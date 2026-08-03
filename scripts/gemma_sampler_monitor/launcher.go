package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	campaignRunID      = "gemma-sampler-v2-single-campaign"
	cuda13BatchName    = "benchmark_gemma_sampler_quality_v2_cuda13.bat"
	launcherLogDirName = "supervisor-logs"
)

func launchedAsSamplerLauncher(arguments []string) bool {
	name := strings.ToLower(filepath.Base(os.Args[0]))
	if name == "gemma-sampler-launcher.exe" || name == "gemma-sampler-launcher" {
		return true
	}
	for _, argument := range arguments {
		if argument == "--launch" {
			return true
		}
	}
	return false
}

func launcherMain(arguments []string) error {
	root, err := launcherRepositoryRoot()
	if err != nil {
		return err
	}
	if containsArgument(arguments, "--verify") {
		fmt.Println("Gemma Sampler 실행 파일 준비 확인 완료")
		fmt.Println("저장소:", root)
		fmt.Println("CUDA13 BAT:", filepath.Join(root, "scripts", cuda13BatchName))
		return nil
	}
	if err := ensureMonitorBuilt(root); err != nil {
		return err
	}

	runRoot := filepath.Join(
		root,
		"banchmark_result_log",
		"managed-runs",
		"10-gemma-translation",
		"gemma-sampler-quality-v2",
		campaignRunID,
	)
	snapshot, snapshotErr := readSnapshot(runRoot, time.Now())
	if snapshotErr == nil {
		switch snapshot.State {
		case "WAITING_FOR_FINAL_JUDGMENT", "WAITING_FOR_JUDGMENT":
			return runMonitor(monitorConfig{
				runRoot:          runRoot,
				pollInterval:     defaultPollInterval,
				gpuInterval:      defaultGPUInterval,
				exitOnCompletion: true,
				exitDelay:        defaultExitDelay,
			})
		case "FAILED_CLOSED", "RELEASE_FAILED":
			return fmt.Errorf("campaign is fail-closed (%s); inspect the saved runner log before retrying", snapshot.State)
		case "RUNNING_JOINT", "RUNNING_MIN_P", "VALIDATING_REUSE", "RELEASING":
			if snapshot.WorkerPID > 0 && windowsCampaignWorkerAlive(snapshot.WorkerPID) {
				return runMonitor(monitorConfig{
					runRoot:          runRoot,
					pollInterval:     defaultPollInterval,
					gpuInterval:      defaultGPUInterval,
					exitOnCompletion: true,
					exitDelay:        defaultExitDelay,
				})
			}
		}
	}

	if err := launchCampaignBatch(root); err != nil {
		return err
	}
	return runMonitor(monitorConfig{
		runRoot:          runRoot,
		pollInterval:     defaultPollInterval,
		gpuInterval:      defaultGPUInterval,
		exitOnCompletion: true,
		exitDelay:        defaultExitDelay,
	})
}

func ensureMonitorBuilt(repositoryRoot string) error {
	buildPath := filepath.Join(repositoryRoot, "scripts", "build_gemma_sampler_monitor.bat")
	if info, err := os.Stat(buildPath); err != nil || info.IsDir() {
		return fmt.Errorf("Gemma monitor build BAT is unavailable")
	}
	logDir, err := launcherLogDirectory(repositoryRoot)
	if err != nil {
		return err
	}
	logPath := filepath.Join(logDir, "launcher-build-"+time.Now().UTC().Format("20060102T150405Z")+".log")
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("open private monitor build log: %w", err)
	}
	defer logFile.Close()
	command := exec.Command("cmd.exe", "/d", "/s", "/c", fmt.Sprintf(`call "%s" --monitor-only-if-stale`, buildPath))
	command.Dir = repositoryRoot
	command.Stdout = logFile
	command.Stderr = logFile
	if err := command.Run(); err != nil {
		return fmt.Errorf("build current Gemma monitor: %w", err)
	}
	return nil
}

func launcherRepositoryRoot() (string, error) {
	starts := make([]string, 0, 2)
	if executable, err := os.Executable(); err == nil {
		starts = append(starts, filepath.Dir(executable))
	}
	if current, err := os.Getwd(); err == nil {
		starts = append(starts, current)
	}
	for _, start := range starts {
		if root, ok := findRepositoryRoot(start); ok {
			return root, nil
		}
	}
	return "", fmt.Errorf("could not find scripts/%s beside this executable", cuda13BatchName)
}

func findRepositoryRoot(start string) (string, bool) {
	current, err := filepath.Abs(start)
	if err != nil {
		return "", false
	}
	for depth := 0; depth <= 8; depth++ {
		candidate := filepath.Join(current, "scripts", cuda13BatchName)
		if info, statErr := os.Stat(candidate); statErr == nil && !info.IsDir() {
			return current, true
		}
		parent := filepath.Dir(current)
		if parent == current {
			break
		}
		current = parent
	}
	return "", false
}

func launchCampaignBatch(repositoryRoot string) error {
	batchPath := filepath.Join(repositoryRoot, "scripts", cuda13BatchName)
	if info, err := os.Stat(batchPath); err != nil || info.IsDir() {
		return fmt.Errorf("CUDA13 campaign BAT is unavailable")
	}
	logDir, err := launcherLogDirectory(repositoryRoot)
	if err != nil {
		return err
	}
	logPath := filepath.Join(logDir, "launcher-"+time.Now().UTC().Format("20060102T150405Z")+".log")
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("open private launcher log: %w", err)
	}
	command := exec.Command("cmd.exe", "/d", "/s", "/c", fmt.Sprintf(`call "%s"`, batchPath))
	command.Dir = repositoryRoot
	command.Stdout = logFile
	command.Stderr = logFile
	command.Env = append(
		os.Environ(),
		"SAMPLER_LAUNCHED_BY_EXE=1",
		"SAMPLER_RUN_ID="+campaignRunID,
	)
	if err := command.Start(); err != nil {
		_ = logFile.Close()
		return fmt.Errorf("start CUDA13 campaign BAT: %w", err)
	}
	go func() {
		_ = command.Wait()
		_ = logFile.Close()
	}()
	return nil
}

func launcherLogDirectory(repositoryRoot string) (string, error) {
	logDir := filepath.Join(
		repositoryRoot,
		"banchmark_result_log",
		"managed-runs",
		"10-gemma-translation",
		"gemma-sampler-quality-v2",
		launcherLogDirName,
	)
	if err := os.MkdirAll(logDir, 0o700); err != nil {
		return "", fmt.Errorf("create private launcher log directory: %w", err)
	}
	return logDir, nil
}

func windowsCampaignWorkerAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	script := fmt.Sprintf("$p = Get-CimInstance -ClassName Win32_Process -Filter 'ProcessId = %d' -ErrorAction SilentlyContinue; if ($null -eq $p -or $p.Name -notmatch '^python(?:\\.exe)?$' -or $p.CommandLine -notlike '*benchmark_gemma_sampler_quality_v2.py*' -or $p.CommandLine -notlike '*run-campaign*') { exit 1 }; exit 0", pid)
	if err := exec.Command("powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script).Run(); err == nil {
		return true
	}
	command := exec.Command("tasklist.exe", "/fi", "PID eq "+strconv.Itoa(pid), "/fo", "csv", "/nh")
	output, err := command.Output()
	if err != nil {
		return false
	}
	return strings.Contains(strings.ToLower(string(output)), "python.exe") && strings.Contains(string(output), strconv.Itoa(pid))
}

func containsArgument(arguments []string, wanted string) bool {
	for _, argument := range arguments {
		if argument == wanted {
			return true
		}
	}
	return false
}
