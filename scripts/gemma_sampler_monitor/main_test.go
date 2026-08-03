package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestReadSnapshotUsesRecentCompletionWindowForETA(t *testing.T) {
	root := t.TempDir()
	now := time.Date(2026, time.August, 3, 14, 30, 0, 0, time.UTC)
	progress := fmt.Sprintf(
		`{"state":"RUNNING","phase":"temperature","completed_logical_slots":100,"phase_expected_logical_slots":400,"updated_utc":%q}`,
		now.Add(-5*time.Second).Format(time.RFC3339),
	)
	if err := os.WriteFile(filepath.Join(root, "progress.json"), []byte(progress), 0o600); err != nil {
		t.Fatal(err)
	}
	lines := make([]string, 0, 120)
	for index := 0; index < 120; index++ {
		stamp := now.Add(-5*time.Second - time.Duration(119-index)*2*time.Second)
		lines = append(lines, fmt.Sprintf(`{"entry":{"recorded_utc":%q}}`, stamp.Format(time.RFC3339)))
	}
	if err := os.WriteFile(filepath.Join(root, "completion-index.jsonl"), []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	snapshot, err := readSnapshot(root, now)
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.State != "RUNNING" || snapshot.Phase != "temperature" {
		t.Fatalf("unexpected snapshot identity: %#v", snapshot)
	}
	if snapshot.Completed != 100 || snapshot.Expected != 400 {
		t.Fatalf("unexpected progress: %#v", snapshot)
	}
	if snapshot.Estimate.Status != "stable" {
		t.Fatalf("expected stable ETA, got %#v", snapshot.Estimate)
	}
	if math.Abs(snapshot.Estimate.RatePerMinute-30) > 0.01 {
		t.Fatalf("expected 30 slots/min, got %.3f", snapshot.Estimate.RatePerMinute)
	}
	if snapshot.Estimate.Remaining != 10*time.Minute {
		t.Fatalf("expected 10m remaining, got %s", snapshot.Estimate.Remaining)
	}
}

func TestReadSnapshotResolvesManagedRunArtifactsDirectory(t *testing.T) {
	runRoot := t.TempDir()
	artifactRoot := filepath.Join(runRoot, "artifacts")
	if err := os.Mkdir(artifactRoot, 0o700); err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, time.August, 3, 14, 30, 0, 0, time.UTC)
	progress := fmt.Sprintf(
		`{"state":"WAITING_FOR_JUDGMENT","phase":"temperature","completed_logical_slots":4,"updated_utc":%q}`,
		now.Format(time.RFC3339),
	)
	if err := os.WriteFile(filepath.Join(artifactRoot, "progress.json"), []byte(progress), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(runRoot, "artifact-manifest.json"), []byte(`{"status":"completed"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(artifactRoot, "phase-status"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(artifactRoot, "phase-status", "temperature.json"), []byte(`{"expected_logical_slots":4}`), 0o600); err != nil {
		t.Fatal(err)
	}

	snapshot, err := readSnapshot(runRoot, now)
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.State != "WAITING_FOR_JUDGMENT" || snapshot.Completed != 4 || snapshot.Expected != 4 {
		t.Fatalf("managed artifacts were not selected: %#v", snapshot)
	}
	if snapshot.ManifestStatus != "completed" {
		t.Fatalf("managed manifest was not selected: %#v", snapshot)
	}
}

func TestCampaignSnapshotShowsNewAndReusedEvidenceWithBlendedETA(t *testing.T) {
	root := t.TempDir()
	now := time.Date(2026, time.August, 3, 14, 30, 0, 0, time.UTC)
	progress := fmt.Sprintf(
		`{"state":"RUNNING_JOINT","phase":"joint_top_p_top_k","campaign_completed_logical_slots":500,"campaign_expected_logical_slots":124280,"reused_logical_slots":9560,"evidence_expected_logical_slots":133840,"stage_completed_logical_slots":500,"stage_expected_logical_slots":105160,"case_count":478,"current_sampler":{"temperature":0.7,"top_p":1,"top_k":0,"min_p":0},"current_seed":20260802,"current_case_position":22,"attempt_counts":{"valid":500,"retry":2,"timeout":1,"indeterminate":3},"active_elapsed_seconds":1000,"initial_eta_seconds":102544,"initial_eta_low_seconds":98400,"initial_eta_high_seconds":119340,"worker_pid":1234,"updated_utc":%q}`,
		now.Add(-5*time.Second).Format(time.RFC3339),
	)
	if err := os.WriteFile(filepath.Join(root, "progress.json"), []byte(progress), 0o600); err != nil {
		t.Fatal(err)
	}
	lines := make([]string, 0, 120)
	for index := 0; index < 120; index++ {
		stamp := now.Add(-5*time.Second - time.Duration(119-index)*2*time.Second)
		lines = append(lines, fmt.Sprintf(`{"recorded_utc":%q}`, stamp.Format(time.RFC3339)))
	}
	if err := os.WriteFile(filepath.Join(root, "completion-index.jsonl"), []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	snapshot, err := readSnapshot(root, now)
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Completed != 500 || snapshot.Expected != 124280 || snapshot.Reused != 9560 || snapshot.EvidenceExpected != 133840 {
		t.Fatalf("campaign evidence counters were not retained: %#v", snapshot)
	}
	if snapshot.StageExpected != 105160 || snapshot.CurrentSampler.TopK != 0 || snapshot.CurrentSeed != 20260802 {
		t.Fatalf("campaign sampler context was not retained: %#v", snapshot)
	}
	if snapshot.Estimate.Status != "stable" || snapshot.Estimate.RatePerMinute <= 0 {
		t.Fatalf("campaign ETA did not blend live data: %#v", snapshot.Estimate)
	}
	if snapshot.Estimate.RecentComplete.IsZero() {
		t.Fatalf("campaign snapshot did not retain the last completion timestamp: %#v", snapshot.Estimate)
	}
}

func TestEstimateCampaignETASeparatesBackoffFromActiveETA(t *testing.T) {
	now := time.Date(2026, time.August, 3, 14, 30, 0, 0, time.UTC)
	timestamps := make([]time.Time, 0, 120)
	for index := 0; index < 120; index++ {
		timestamps = append(timestamps, now.Add(-5*time.Second-time.Duration(119-index)*2*time.Second))
	}
	estimate := estimateCampaignETA(
		timestamps,
		500,
		124280,
		1000,
		100,
		102544,
		98400,
		119340,
		now.Add(-5*time.Second),
		now,
	)
	if estimate.Status != "stable" || estimate.ActiveRemaining <= 0 || estimate.ExpectedBackoff <= 0 {
		t.Fatalf("campaign ETA did not expose active and backoff portions: %#v", estimate)
	}
	if estimate.Remaining != estimate.ActiveRemaining+estimate.ExpectedBackoff {
		t.Fatalf("wall ETA did not include the expected backoff: %#v", estimate)
	}
}

func TestFindRepositoryRootFindsCuda13BatchFromLauncherLocation(t *testing.T) {
	root := t.TempDir()
	scripts := filepath.Join(root, "scripts")
	if err := os.Mkdir(scripts, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(scripts, cuda13BatchName), []byte("@echo off\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	launcherDir := filepath.Join(root, "private-artifacts", "tools")
	if err := os.MkdirAll(launcherDir, 0o700); err != nil {
		t.Fatal(err)
	}

	found, ok := findRepositoryRoot(launcherDir)
	if !ok || found != root {
		t.Fatalf("launcher could not resolve its repository root: found=%q ok=%v", found, ok)
	}
}

func TestEstimateCompletionETAIgnoresInterruptedIdleGap(t *testing.T) {
	now := time.Date(2026, time.August, 3, 14, 30, 0, 0, time.UTC)
	timestamps := []time.Time{now.Add(-4 * time.Hour), now.Add(-4*time.Hour + 2*time.Second)}
	for index := 0; index < 120; index++ {
		timestamps = append(timestamps, now.Add(-time.Duration(119-index)*2*time.Second))
	}
	estimate := estimateCompletionETA(timestamps, 300, now, now)
	if estimate.Status != "stable" {
		t.Fatalf("expected stable ETA after recent continuous work, got %#v", estimate)
	}
	if math.Abs(estimate.RatePerMinute-30) > 0.01 {
		t.Fatalf("long interruption distorted the recent rate: %.3f", estimate.RatePerMinute)
	}
	if estimate.Observation > 5*time.Minute {
		t.Fatalf("long interruption leaked into observation window: %s", estimate.Observation)
	}
}

func TestParseMonitorConfigRequiresAValidRunRootAndIntervals(t *testing.T) {
	config, err := parseMonitorConfig([]string{"--run-root", ".", "--poll-interval", "1s", "--no-gpu"})
	if err != nil {
		t.Fatal(err)
	}
	if config.runRoot == "." || !config.disableGPU {
		t.Fatalf("unexpected config: %#v", config)
	}
	if _, err := parseMonitorConfig([]string{"--run-root", ".", "--poll-interval", "100ms"}); err == nil {
		t.Fatal("expected invalid poll interval to fail")
	}
}

func TestProgressReaderClosesBeforeTheWriterAtomicallyReplacesTheSnapshot(t *testing.T) {
	root := t.TempDir()
	destination := filepath.Join(root, "progress.json")
	replacement := filepath.Join(root, "replacement.json")
	if err := os.WriteFile(destination, []byte(`{"state":"old"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := readSharedFile(destination); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(replacement, []byte(`{"state":"new"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(replacement, destination); err != nil {
		t.Fatalf("monitor left a progress handle open after a bounded read: %v", err)
	}
}

func TestFormatGPUUsesAStableTwoLineLayoutPerDevice(t *testing.T) {
	formatted := formatGPU(
		[]GPUStat{{
			Index:       "0",
			Name:        "NVIDIA GeForce RTX 4070 SUPER",
			MemoryUsed:  774,
			MemoryTotal: 12282,
			Utilization: 19,
			Temperature: 36,
		}},
		nil,
		false,
	)
	if !strings.Contains(formatted, "\n") || !strings.Contains(formatted, "VRAM 774 / 12282 MiB") {
		t.Fatalf("GPU details are not intentionally laid out: %q", formatted)
	}
}
