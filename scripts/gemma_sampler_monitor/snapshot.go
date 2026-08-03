package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	completionTailBytes = 1024 * 1024
	completionSampleMax = 360
	progressStaleAfter  = 90 * time.Second
	continuityGap       = 90 * time.Second
)

type progressRecord struct {
	State      string `json:"state"`
	Phase      string `json:"phase"`
	Completed  int    `json:"completed_logical_slots"`
	Expected   int    `json:"phase_expected_logical_slots"`
	UpdatedUTC string `json:"updated_utc"`
	Schema     string `json:"schema_version"`
}

type manifestRecord struct {
	Status string `json:"status"`
}

type completionIndexRecord struct {
	RecordedUTC string `json:"recorded_utc"`
}

type phaseStatusRecord struct {
	Expected int `json:"expected_logical_slots"`
}

type ETAEstimate struct {
	Status         string
	RatePerMinute  float64
	Remaining      time.Duration
	FinishAt       time.Time
	SampleCount    int
	Observation    time.Duration
	RecentComplete time.Time
}

type Snapshot struct {
	RunRoot        string
	State          string
	Phase          string
	Completed      int
	Expected       int
	UpdatedAt      time.Time
	ManifestStatus string
	Estimate       ETAEstimate
	ReadAt         time.Time
}

type GPUStat struct {
	Index       string
	Name        string
	MemoryUsed  int
	MemoryTotal int
	Utilization int
	Temperature int
}

func readSnapshot(runRoot string, now time.Time) (Snapshot, error) {
	snapshot := Snapshot{RunRoot: runRoot, ReadAt: now}
	artifactRoot := resolveArtifactRoot(runRoot)
	progressPath := filepath.Join(artifactRoot, "progress.json")
	data, err := readSharedFile(progressPath)
	if err != nil {
		return snapshot, err
	}
	var progress progressRecord
	if err := json.Unmarshal(data, &progress); err != nil {
		return snapshot, fmt.Errorf("decode progress snapshot: %w", err)
	}
	if progress.Completed < 0 || progress.Expected < 0 || (progress.Expected > 0 && progress.Completed > progress.Expected) {
		return snapshot, fmt.Errorf("progress snapshot has impossible completion counts")
	}
	snapshot.State = strings.TrimSpace(progress.State)
	snapshot.Phase = strings.TrimSpace(progress.Phase)
	snapshot.Completed = progress.Completed
	snapshot.Expected = progress.Expected
	snapshot.UpdatedAt = parseUTCTimestamp(progress.UpdatedUTC)
	if snapshot.Expected == 0 && snapshot.Phase != "" {
		snapshot.Expected = readPhaseExpected(artifactRoot, snapshot.Phase)
	}
	snapshot.ManifestStatus = readManifestStatus(filepath.Join(managedRunRoot(runRoot), "artifact-manifest.json"))

	timestamps, indexErr := readRecentCompletionTimes(filepath.Join(artifactRoot, "completion-index.jsonl"), completionSampleMax)
	if indexErr == nil {
		snapshot.Estimate = estimateCompletionETA(
			timestamps,
			maxInt(snapshot.Expected-snapshot.Completed, 0),
			snapshot.UpdatedAt,
			now,
		)
	} else {
		snapshot.Estimate = ETAEstimate{Status: "measuring"}
	}
	return snapshot, nil
}

func readPhaseExpected(artifactRoot, phase string) int {
	if !knownPhase(phase) {
		return 0
	}
	data, err := readSharedFile(filepath.Join(artifactRoot, "phase-status", phase+".json"))
	if err != nil {
		return 0
	}
	var status phaseStatusRecord
	if json.Unmarshal(data, &status) != nil || status.Expected < 0 {
		return 0
	}
	return status.Expected
}

func knownPhase(phase string) bool {
	switch phase {
	case "temperature", "joint_top_p_top_k", "min_p":
		return true
	default:
		return false
	}
}

// The BAT starts the monitor before the managed run exists.  Once the harness
// creates its artifacts directory, resolve it automatically while still
// accepting an artifacts directory passed directly for diagnostics.
func resolveArtifactRoot(runRoot string) string {
	candidate := filepath.Join(runRoot, "artifacts")
	info, err := os.Stat(candidate)
	if err == nil && info.IsDir() {
		return candidate
	}
	return runRoot
}

func managedRunRoot(runRoot string) string {
	if strings.EqualFold(filepath.Base(filepath.Clean(runRoot)), "artifacts") {
		return filepath.Dir(runRoot)
	}
	return runRoot
}

func readManifestStatus(path string) string {
	data, err := readSharedFile(path)
	if err != nil {
		return ""
	}
	var manifest manifestRecord
	if json.Unmarshal(data, &manifest) != nil {
		return ""
	}
	return strings.TrimSpace(manifest.Status)
}

func readSharedFile(path string) ([]byte, error) {
	// Do not tail or retain a handle to the atomic JSON snapshots.  On Windows a
	// replace can briefly race a bounded read, so the writer's existing atomic
	// retry contract handles that short interval; this monitor always closes its
	// handle before returning to Bubble Tea.
	var lastErr error
	for attempt := 0; attempt < 4; attempt++ {
		file, err := os.Open(path)
		if err == nil {
			data, readErr := io.ReadAll(file)
			closeErr := file.Close()
			if readErr == nil && closeErr == nil {
				return data, nil
			}
			if readErr != nil {
				lastErr = readErr
			} else {
				lastErr = closeErr
			}
		} else {
			lastErr = err
		}
		if attempt < 3 {
			time.Sleep(time.Duration(attempt+1) * 25 * time.Millisecond)
		}
	}
	return nil, lastErr
}

func readRecentCompletionTimes(path string, limit int) ([]time.Time, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, err
	}
	start := maxInt64(0, info.Size()-completionTailBytes)
	if _, err := file.Seek(start, io.SeekStart); err != nil {
		return nil, err
	}
	reader := bufio.NewReader(file)
	if start > 0 {
		_, _ = reader.ReadString('\n')
	}
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 4096), 1024*1024)
	timestamps := make([]time.Time, 0, limit)
	for scanner.Scan() {
		var record completionIndexRecord
		if json.Unmarshal(scanner.Bytes(), &record) != nil {
			// The active writer can leave only its final append incomplete.  It
			// will become a valid line on the following refresh.
			continue
		}
		if parsed := parseUTCTimestamp(record.RecordedUTC); !parsed.IsZero() {
			timestamps = append(timestamps, parsed)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(timestamps) > limit {
		timestamps = timestamps[len(timestamps)-limit:]
	}
	sort.Slice(timestamps, func(left, right int) bool { return timestamps[left].Before(timestamps[right]) })
	return timestamps, nil
}

func estimateCompletionETA(timestamps []time.Time, remaining int, updatedAt, now time.Time) ETAEstimate {
	if remaining == 0 {
		return ETAEstimate{Status: "complete"}
	}
	if updatedAt.IsZero() || now.Sub(updatedAt) > progressStaleAfter {
		return ETAEstimate{Status: "stalled"}
	}
	if len(timestamps) < 4 {
		return ETAEstimate{Status: "measuring", SampleCount: len(timestamps)}
	}
	start := len(timestamps) - 1
	for start > 0 && timestamps[start].Sub(timestamps[start-1]) <= continuityGap {
		start--
	}
	contiguous := timestamps[start:]
	windowSizes := []int{30, 90, 240}
	rates := make([]float64, 0, len(windowSizes))
	usedCount := 0
	for _, requested := range windowSizes {
		count := minInt(requested, len(contiguous))
		if count < 4 {
			continue
		}
		window := contiguous[len(contiguous)-count:]
		span := window[len(window)-1].Sub(window[0])
		if span < 10*time.Second {
			continue
		}
		rate := float64(count-1) / span.Minutes()
		if rate > 0 && !math.IsInf(rate, 0) && !math.IsNaN(rate) {
			rates = append(rates, rate)
			usedCount = maxInt(usedCount, count)
		}
	}
	if len(rates) == 0 {
		return ETAEstimate{Status: "measuring", SampleCount: len(contiguous)}
	}
	sort.Float64s(rates)
	rate := rates[len(rates)/2]
	span := contiguous[len(contiguous)-1].Sub(contiguous[0])
	remainingDuration := time.Duration((float64(remaining) / rate) * float64(time.Minute))
	status := "measuring"
	if usedCount >= 90 && span >= time.Minute {
		status = "stable"
	}
	return ETAEstimate{
		Status:         status,
		RatePerMinute:  rate,
		Remaining:      remainingDuration,
		FinishAt:       now.Add(remainingDuration),
		SampleCount:    usedCount,
		Observation:    span,
		RecentComplete: contiguous[len(contiguous)-1],
	}
}

func readGPUStats() ([]GPUStat, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	command := exec.CommandContext(
		ctx,
		"nvidia-smi.exe",
		"--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
		"--format=csv,noheader,nounits",
	)
	output, err := command.Output()
	if err != nil {
		return nil, err
	}
	stats := make([]GPUStat, 0)
	for _, line := range strings.Split(strings.TrimSpace(string(output)), "\n") {
		fields := strings.Split(line, ",")
		if len(fields) != 6 {
			continue
		}
		values := make([]int, 0, 3)
		for _, raw := range fields[2:] {
			value, parseErr := strconv.Atoi(strings.TrimSpace(raw))
			if parseErr != nil {
				values = nil
				break
			}
			values = append(values, value)
		}
		if len(values) != 4 {
			continue
		}
		stats = append(stats, GPUStat{
			Index:       strings.TrimSpace(fields[0]),
			Name:        strings.TrimSpace(fields[1]),
			MemoryUsed:  values[0],
			MemoryTotal: values[1],
			Utilization: values[2],
			Temperature: values[3],
		})
	}
	if len(stats) == 0 {
		return nil, fmt.Errorf("nvidia-smi produced no parseable GPU rows")
	}
	return stats, nil
}

func parseUTCTimestamp(raw string) time.Time {
	parsed, err := time.Parse(time.RFC3339, strings.TrimSpace(raw))
	if err != nil {
		return time.Time{}
	}
	return parsed
}

func maxInt(left, right int) int {
	if left > right {
		return left
	}
	return right
}

func minInt(left, right int) int {
	if left < right {
		return left
	}
	return right
}

func maxInt64(left, right int64) int64 {
	if left > right {
		return left
	}
	return right
}
