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
	State             string        `json:"state"`
	Phase             string        `json:"phase"`
	Completed         int           `json:"completed_logical_slots"`
	Expected          int           `json:"phase_expected_logical_slots"`
	CampaignCompleted int           `json:"campaign_completed_logical_slots"`
	CampaignExpected  int           `json:"campaign_expected_logical_slots"`
	Reused            int           `json:"reused_logical_slots"`
	EvidenceExpected  int           `json:"evidence_expected_logical_slots"`
	StageCompleted    int           `json:"stage_completed_logical_slots"`
	StageExpected     int           `json:"stage_expected_logical_slots"`
	CaseCount         int           `json:"case_count"`
	CurrentSampler    samplerRecord `json:"current_sampler"`
	CurrentArm        string        `json:"current_arm_key"`
	CurrentSeed       int           `json:"current_seed"`
	CurrentCase       int           `json:"current_case_position"`
	AttemptCounts     attemptRecord `json:"attempt_counts"`
	ActiveSeconds     float64       `json:"active_elapsed_seconds"`
	BackoffSeconds    float64       `json:"backoff_elapsed_seconds"`
	InitialETASeconds int           `json:"initial_eta_seconds"`
	InitialETALow     int           `json:"initial_eta_low_seconds"`
	InitialETAHigh    int           `json:"initial_eta_high_seconds"`
	Detail            string        `json:"detail"`
	WorkerPID         int           `json:"worker_pid"`
	UpdatedUTC        string        `json:"updated_utc"`
	Schema            string        `json:"schema_version"`
}

type samplerRecord struct {
	Temperature float64 `json:"temperature"`
	TopP        float64 `json:"top_p"`
	TopK        int     `json:"top_k"`
	MinP        float64 `json:"min_p"`
}

type attemptRecord struct {
	Valid         int `json:"valid"`
	Retry         int `json:"retry"`
	Timeout       int `json:"timeout"`
	Indeterminate int `json:"indeterminate"`
}

type manifestRecord struct {
	Status string `json:"status"`
}

type completionIndexRecord struct {
	RecordedUTC string `json:"recorded_utc"`
	Entry       struct {
		RecordedUTC string `json:"recorded_utc"`
	} `json:"entry"`
}

type phaseStatusRecord struct {
	Expected int `json:"expected_logical_slots"`
}

type ETAEstimate struct {
	Status          string
	RatePerMinute   float64
	Remaining       time.Duration
	FinishAt        time.Time
	SampleCount     int
	Observation     time.Duration
	RecentComplete  time.Time
	LowRemaining    time.Duration
	HighRemaining   time.Duration
	ActiveRemaining time.Duration
	ExpectedBackoff time.Duration
}

type Snapshot struct {
	RunRoot          string
	State            string
	Phase            string
	Completed        int
	Expected         int
	Reused           int
	EvidenceExpected int
	StageCompleted   int
	StageExpected    int
	CaseCount        int
	CurrentSampler   samplerRecord
	CurrentArm       string
	CurrentSeed      int
	CurrentCase      int
	Attempts         attemptRecord
	ActiveSeconds    float64
	BackoffSeconds   float64
	Detail           string
	WorkerPID        int
	UpdatedAt        time.Time
	ManifestStatus   string
	Estimate         ETAEstimate
	ReadAt           time.Time
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
	if progress.CampaignCompleted < 0 || progress.CampaignExpected < 0 || (progress.CampaignExpected > 0 && progress.CampaignCompleted > progress.CampaignExpected) || progress.Reused < 0 || progress.EvidenceExpected < 0 || (progress.EvidenceExpected > 0 && progress.CampaignCompleted+progress.Reused > progress.EvidenceExpected) || progress.StageCompleted < 0 || progress.StageExpected < 0 || (progress.StageExpected > 0 && progress.StageCompleted > progress.StageExpected) {
		return snapshot, fmt.Errorf("campaign progress snapshot has impossible completion counts")
	}
	snapshot.State = strings.TrimSpace(progress.State)
	snapshot.Phase = strings.TrimSpace(progress.Phase)
	snapshot.Completed = progress.Completed
	snapshot.Expected = progress.Expected
	if progress.CampaignExpected > 0 {
		snapshot.Completed = progress.CampaignCompleted
		snapshot.Expected = progress.CampaignExpected
	}
	snapshot.Reused = progress.Reused
	snapshot.EvidenceExpected = progress.EvidenceExpected
	snapshot.StageCompleted = progress.StageCompleted
	snapshot.StageExpected = progress.StageExpected
	snapshot.CaseCount = progress.CaseCount
	snapshot.CurrentSampler = progress.CurrentSampler
	snapshot.CurrentArm = strings.TrimSpace(progress.CurrentArm)
	snapshot.CurrentSeed = progress.CurrentSeed
	snapshot.CurrentCase = progress.CurrentCase
	snapshot.Attempts = progress.AttemptCounts
	snapshot.ActiveSeconds = progress.ActiveSeconds
	snapshot.BackoffSeconds = progress.BackoffSeconds
	snapshot.Detail = strings.TrimSpace(progress.Detail)
	snapshot.WorkerPID = progress.WorkerPID
	snapshot.UpdatedAt = parseUTCTimestamp(progress.UpdatedUTC)
	if snapshot.Expected == 0 && snapshot.Phase != "" {
		snapshot.Expected = readPhaseExpected(artifactRoot, snapshot.Phase)
	}
	snapshot.ManifestStatus = readManifestStatus(filepath.Join(managedRunRoot(runRoot), "artifact-manifest.json"))

	timestamps, indexErr := readRecentCompletionTimes(filepath.Join(artifactRoot, "completion-index.jsonl"), completionSampleMax)
	if indexErr == nil {
		if progress.CampaignExpected > 0 {
			snapshot.Estimate = estimateCampaignETA(
				timestamps,
				snapshot.Completed,
				snapshot.Expected,
				snapshot.ActiveSeconds,
				snapshot.BackoffSeconds,
				progress.InitialETASeconds,
				progress.InitialETALow,
				progress.InitialETAHigh,
				snapshot.UpdatedAt,
				now,
			)
		} else {
			snapshot.Estimate = estimateCompletionETA(
				timestamps,
				maxInt(snapshot.Expected-snapshot.Completed, 0),
				snapshot.UpdatedAt,
				now,
			)
		}
	} else if progress.CampaignExpected > 0 {
		snapshot.Estimate = estimateCampaignETA(
			nil,
			snapshot.Completed,
			snapshot.Expected,
			snapshot.ActiveSeconds,
			snapshot.BackoffSeconds,
			progress.InitialETASeconds,
			progress.InitialETALow,
			progress.InitialETAHigh,
			snapshot.UpdatedAt,
			now,
		)
	} else {
		snapshot.Estimate = ETAEstimate{Status: "measuring"}
	}
	return snapshot, nil
}

func estimateCampaignETA(
	timestamps []time.Time,
	completed int,
	expected int,
	activeSeconds float64,
	backoffSeconds float64,
	initialSeconds int,
	initialLow int,
	initialHigh int,
	updatedAt time.Time,
	now time.Time,
) ETAEstimate {
	remaining := maxInt(expected-completed, 0)
	if remaining == 0 {
		return ETAEstimate{Status: "complete"}
	}
	if updatedAt.IsZero() || now.Sub(updatedAt) > progressStaleAfter {
		return ETAEstimate{Status: "stalled"}
	}
	if completed < 500 || initialSeconds <= 0 {
		base := time.Duration(initialSeconds) * time.Second
		if initialSeconds <= 0 {
			base = 0
		}
		ratio := float64(remaining) / math.Max(float64(expected), 1)
		remainingDuration := time.Duration(float64(base) * ratio)
		recentComplete := time.Time{}
		if len(timestamps) > 0 {
			recentComplete = timestamps[len(timestamps)-1]
		}
		return ETAEstimate{
			Status:          "baseline",
			Remaining:       remainingDuration,
			ActiveRemaining: remainingDuration,
			FinishAt:        now.Add(remainingDuration),
			SampleCount:     completed,
			RecentComplete:  recentComplete,
			LowRemaining:    time.Duration(float64(time.Duration(initialLow)*time.Second) * ratio),
			HighRemaining:   time.Duration(float64(time.Duration(initialHigh)*time.Second) * ratio),
		}
	}
	recent := estimateCompletionETA(timestamps, remaining, updatedAt, now)
	if recent.RatePerMinute <= 0 || activeSeconds <= 0 {
		return recent
	}
	overallRate := float64(completed) / (activeSeconds / 60.0)
	if overallRate <= 0 || math.IsInf(overallRate, 0) || math.IsNaN(overallRate) {
		return recent
	}
	backoffRatio := math.Max(0, backoffSeconds) / activeSeconds
	recentActiveRate := recent.RatePerMinute * (1 + backoffRatio)
	combinedRate := recentActiveRate*0.7 + overallRate*0.3
	activeRemaining := time.Duration((float64(remaining) / combinedRate) * float64(time.Minute))
	expectedBackoff := time.Duration(float64(activeRemaining) * backoffRatio)
	remainingDuration := activeRemaining + expectedBackoff
	recent.Status = "stable"
	recent.RatePerMinute = combinedRate
	recent.Remaining = remainingDuration
	recent.ActiveRemaining = activeRemaining
	recent.ExpectedBackoff = expectedBackoff
	recent.FinishAt = now.Add(remainingDuration)
	return recent
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
		recordedUTC := record.RecordedUTC
		if recordedUTC == "" {
			recordedUTC = record.Entry.RecordedUTC
		}
		if parsed := parseUTCTimestamp(recordedUTC); !parsed.IsZero() {
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
