package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

const (
	defaultPollInterval = time.Second
	defaultGPUInterval  = 5 * time.Second
	defaultExitDelay    = 12 * time.Second
)

type monitorConfig struct {
	runRoot          string
	pollInterval     time.Duration
	gpuInterval      time.Duration
	exitOnCompletion bool
	exitDelay        time.Duration
	disableGPU       bool
}

type snapshotMessage struct {
	snapshot Snapshot
	err      error
}

type gpuMessage struct {
	stats []GPUStat
	err   error
}

type tickMessage time.Time
type terminalExitMessage struct{}

type monitorModel struct {
	config                monitorConfig
	snapshot              Snapshot
	snapshotErr           error
	gpu                   []GPUStat
	gpuErr                error
	lastGPURead           time.Time
	now                   time.Time
	width                 int
	terminalExitScheduled bool
	terminalExitAt        time.Time
}

var (
	titleStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("230")).Background(lipgloss.Color("57")).Padding(0, 1)
	labelStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("69"))
	valueStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("252"))
	dimStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("243"))
	okStyle    = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("42"))
	warnStyle  = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("214"))
	badStyle   = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("203"))
	panelStyle = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color("63")).Padding(0, 1)
)

func main() {
	if launchedAsSamplerLauncher(os.Args[1:]) {
		if err := launcherMain(os.Args[1:]); err != nil {
			fmt.Fprintln(os.Stderr, "gemma-sampler-launcher:", err)
			os.Exit(2)
		}
		return
	}
	config, err := parseMonitorConfig(os.Args[1:])
	if err != nil {
		fmt.Fprintln(os.Stderr, "gemma-monitor:", err)
		os.Exit(2)
	}
	if err := runMonitor(config); err != nil {
		fmt.Fprintln(os.Stderr, "gemma-monitor:", err)
		os.Exit(1)
	}
}

func runMonitor(config monitorConfig) error {
	program := tea.NewProgram(newMonitorModel(config), tea.WithAltScreen())
	if _, err := program.Run(); err != nil {
		return err
	}
	return nil
}

func parseMonitorConfig(arguments []string) (monitorConfig, error) {
	flags := flag.NewFlagSet("gemma-monitor", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	var config monitorConfig
	flags.StringVar(&config.runRoot, "run-root", "", "private sampler managed-run root or artifacts directory")
	flags.DurationVar(&config.pollInterval, "poll-interval", defaultPollInterval, "progress refresh interval")
	flags.DurationVar(&config.gpuInterval, "gpu-interval", defaultGPUInterval, "NVIDIA telemetry refresh interval")
	flags.BoolVar(&config.exitOnCompletion, "exit-on-completion", false, "close after WAITING_FOR_JUDGMENT")
	flags.DurationVar(&config.exitDelay, "completion-exit-delay", defaultExitDelay, "delay before closing after normal completion")
	flags.BoolVar(&config.disableGPU, "no-gpu", false, "do not query NVIDIA telemetry")
	if err := flags.Parse(arguments); err != nil {
		return monitorConfig{}, err
	}
	if strings.TrimSpace(config.runRoot) == "" {
		return monitorConfig{}, errors.New("--run-root is required")
	}
	resolved, err := filepath.Abs(config.runRoot)
	if err != nil {
		return monitorConfig{}, fmt.Errorf("resolve --run-root: %w", err)
	}
	config.runRoot = filepath.Clean(resolved)
	if config.pollInterval < 500*time.Millisecond || config.pollInterval > time.Minute {
		return monitorConfig{}, errors.New("--poll-interval must be between 500ms and 1m")
	}
	if config.gpuInterval < time.Second || config.gpuInterval > 5*time.Minute {
		return monitorConfig{}, errors.New("--gpu-interval must be between 1s and 5m")
	}
	if config.exitDelay < 0 || config.exitDelay > 5*time.Minute {
		return monitorConfig{}, errors.New("--completion-exit-delay must be between 0s and 5m")
	}
	return config, nil
}

func newMonitorModel(config monitorConfig) monitorModel {
	return monitorModel{config: config, now: time.Now()}
}

func (model monitorModel) Init() tea.Cmd {
	commands := []tea.Cmd{model.snapshotCommand(), tickCommand(model.config.pollInterval)}
	if !model.config.disableGPU {
		commands = append(commands, gpuCommand())
	}
	return tea.Batch(commands...)
}

func (model monitorModel) Update(message tea.Msg) (tea.Model, tea.Cmd) {
	switch typed := message.(type) {
	case tea.KeyMsg:
		switch typed.String() {
		case "q", "ctrl+c", "esc":
			return model, tea.Quit
		}
	case tea.WindowSizeMsg:
		model.width = typed.Width
	case tickMessage:
		model.now = time.Time(typed)
		commands := []tea.Cmd{model.snapshotCommand(), tickCommand(model.config.pollInterval)}
		if !model.config.disableGPU && model.now.Sub(model.lastGPURead) >= model.config.gpuInterval {
			commands = append(commands, gpuCommand())
		}
		return model, tea.Batch(commands...)
	case snapshotMessage:
		model.now = time.Now()
		model.snapshot = typed.snapshot
		model.snapshotErr = typed.err
		if typed.err == nil && isTerminalSuccessState(typed.snapshot.State) && model.config.exitOnCompletion && !model.terminalExitScheduled {
			model.terminalExitScheduled = true
			model.terminalExitAt = model.now.Add(model.config.exitDelay)
			return model, tea.Tick(model.config.exitDelay, func(time.Time) tea.Msg { return terminalExitMessage{} })
		}
		if !isTerminalSuccessState(typed.snapshot.State) {
			model.terminalExitScheduled = false
			model.terminalExitAt = time.Time{}
		}
	case gpuMessage:
		model.lastGPURead = time.Now()
		model.gpu = typed.stats
		model.gpuErr = typed.err
	case terminalExitMessage:
		if model.snapshotErr == nil && isTerminalSuccessState(model.snapshot.State) {
			return model, tea.Quit
		}
	}
	return model, nil
}

func (model monitorModel) View() string {
	width := model.width
	if width <= 0 {
		width = 104
	}
	width = maxInt(72, minInt(width, 128))
	innerWidth := width - 4

	header := titleStyle.Render("GEMMA SAMPLER  |  실시간 실행 모니터 / LIVE RUN MONITOR")
	runLine := dimStyle.Render("run-root (read-only): ") + valueStyle.Render(shortenMiddle(model.config.runRoot, innerWidth-24))
	if model.snapshotErr != nil {
		waiting := panelStyle.Width(innerWidth).Render(strings.Join([]string{
			warnStyle.Render("진행 파일 대기 중 / WAITING FOR RUN ARTIFACTS"),
			"BAT runner가 시작되면 progress.json을 자동으로 읽습니다.",
			dimStyle.Render("모니터는 read-only이며 runner, Docker, GPU 작업을 멈추지 않습니다."),
		}, "\n"))
		return strings.Join([]string{header, runLine, waiting, footerLine()}, "\n\n") + "\n"
	}

	percentage := completionRatio(model.snapshot.Completed, model.snapshot.Expected)
	stagePercentage := completionRatio(model.snapshot.StageCompleted, model.snapshot.StageExpected)
	evidenceCompleted := model.snapshot.Completed + model.snapshot.Reused
	evidencePercentage := completionRatio(evidenceCompleted, model.snapshot.EvidenceExpected)
	progressLines := []string{
		renderProgressBar(percentage, maxInt(30, innerWidth-2)) + " " + fmt.Sprintf("%5.1f%%", percentage*100),
		fmt.Sprintf("%s  %s / %s 완료  |  %s %s", labelStyle.Render("새 실행 / NEW:"), formatCount(model.snapshot.Completed), formatCount(model.snapshot.Expected), labelStyle.Render("남음 / LEFT:"), formatCount(maxInt(model.snapshot.Expected-model.snapshot.Completed, 0))),
	}
	if model.snapshot.EvidenceExpected > 0 {
		progressLines = append(progressLines,
			fmt.Sprintf("%s  r6 재사용 %s  |  전체 증거 %s / %s (%.1f%%)", labelStyle.Render("증거 / EVIDENCE:"), formatCount(model.snapshot.Reused), formatCount(evidenceCompleted), formatCount(model.snapshot.EvidenceExpected), evidencePercentage*100),
		)
	}
	if model.snapshot.StageExpected > 0 {
		progressLines = append(progressLines,
			fmt.Sprintf("%s  %s / %s (%.1f%%)", labelStyle.Render("현재 단계 / STAGE:"), formatCount(model.snapshot.StageCompleted), formatCount(model.snapshot.StageExpected), stagePercentage*100),
		)
	}
	if model.snapshot.CurrentArm != "" {
		progressLines = append(progressLines, samplerLine(model.snapshot))
	}
	phasePanel := panelStyle.Width(innerWidth).Render(strings.Join([]string{
		labelStyle.Render("현재 단계 / CURRENT PHASE") + "  " + phaseLabel(model.snapshot.Phase),
		stateLine(model.snapshot.State, model.snapshot.ManifestStatus),
		strings.Join(progressLines, "\n"),
		phaseDescription(model.snapshot.Phase, model.snapshot.Expected),
	}, "\n"))

	etaPanel := panelStyle.Width(innerWidth).Render(strings.Join([]string{
		labelStyle.Render("속도·예상 종료 / RATE AND ETA"),
		formatETA(model.snapshot.Estimate),
		activityLine(model.snapshot),
		freshnessLine(model.snapshot.UpdatedAt, model.now),
	}, "\n"))

	gpuPanel := panelStyle.Width(innerWidth).Render(strings.Join([]string{
		labelStyle.Render("GPU 텔레메트리 / GPU TELEMETRY"),
		formatGPU(model.gpu, model.gpuErr, model.config.disableGPU),
	}, "\n"))

	checkpointLines := []string{
		labelStyle.Render("체크포인트·로그 / CHECKPOINT AND LOG"),
		freshnessLine(model.snapshot.UpdatedAt, model.now),
		lastCompletionLine(model.snapshot.Estimate),
		dimStyle.Render("runner log: ") + valueStyle.Render(shortenMiddle(supervisorLogPath(model.config.runRoot), innerWidth-14)),
	}
	if model.snapshot.Detail != "" {
		checkpointLines = append(checkpointLines, warnStyle.Render("마지막 상태: ")+valueStyle.Render(shortenMiddle(model.snapshot.Detail, innerWidth-18)))
	}
	if model.snapshot.WorkerPID > 0 {
		checkpointLines = append(checkpointLines, dimStyle.Render(fmt.Sprintf("현재 Python worker PID: %d", model.snapshot.WorkerPID)))
	}
	checkpointPanel := panelStyle.Width(innerWidth).Render(strings.Join(checkpointLines, "\n"))

	terminalLine := ""
	if model.terminalExitScheduled {
		remaining := maxDuration(model.terminalExitAt.Sub(model.now), 0)
		terminalLine = okStyle.Render("정상 실행 종료 / WAITING FOR FINAL JUDGMENT") + "  " + dimStyle.Render("이 창은 "+formatDuration(remaining)+" 후 자동 종료됩니다.")
	}
	parts := []string{header, runLine, phasePanel, etaPanel, gpuPanel, checkpointPanel}
	if terminalLine != "" {
		parts = append(parts, terminalLine)
	}
	parts = append(parts, footerLine())
	return strings.Join(parts, "\n\n") + "\n"
}

func (model monitorModel) snapshotCommand() tea.Cmd {
	runRoot := model.config.runRoot
	return func() tea.Msg {
		snapshot, err := readSnapshot(runRoot, time.Now())
		return snapshotMessage{snapshot: snapshot, err: err}
	}
}

func tickCommand(interval time.Duration) tea.Cmd {
	return tea.Tick(interval, func(when time.Time) tea.Msg { return tickMessage(when) })
}

func gpuCommand() tea.Cmd {
	return func() tea.Msg {
		stats, err := readGPUStats()
		return gpuMessage{stats: stats, err: err}
	}
}

func completionRatio(completed, expected int) float64 {
	if expected <= 0 {
		return 0
	}
	return math.Min(1, math.Max(0, float64(completed)/float64(expected)))
}

func isTerminalSuccessState(state string) bool {
	return state == "WAITING_FOR_JUDGMENT" || state == "WAITING_FOR_FINAL_JUDGMENT"
}

func renderProgressBar(ratio float64, width int) string {
	filled := int(math.Round(ratio * float64(width)))
	filled = minInt(width, maxInt(0, filled))
	return okStyle.Render(strings.Repeat("█", filled)) + dimStyle.Render(strings.Repeat("░", width-filled))
}

func phaseLabel(phase string) string {
	switch phase {
	case "temperature":
		return valueStyle.Render("온도 / Temperature")
	case "joint_top_p_top_k":
		return valueStyle.Render("top-p·top-k 조합 / Joint top-p·top-k")
	case "min_p":
		return valueStyle.Render("min-p / Minimum probability")
	default:
		return valueStyle.Render("준비 중 / Preparing")
	}
}

func phaseDescription(phase string, expected int) string {
	if expected <= 0 {
		return dimStyle.Render("실행 matrix 정보를 읽는 중 / Reading the execution matrix")
	}
	switch phase {
	case "":
		return dimStyle.Render("r6 재사용을 검증하고 Router를 준비 중입니다 / Validating r6 reuse and preparing Router")
	case "temperature":
		return dimStyle.Render("matrix: temperature 0.1–1.0 × 2 seeds × 478 cases = " + formatCount(expected))
	case "joint_top_p_top_k":
		return dimStyle.Render("고정 matrix: 온도 10개 × top-p 4개 × top-k 3개 × 2 seed (r6 기본값 10개 재사용)")
	case "min_p":
		return dimStyle.Render("고정 matrix: 온도 10개 × min-p 0/.05/.10 × 2 seed (0은 joint 결과 재사용)")
	default:
		return dimStyle.Render("matrix slots: " + formatCount(expected))
	}
}

func stateLine(state, manifestStatus string) string {
	label := state
	style := valueStyle
	switch state {
	case "RUNNING", "RUNNING_JOINT", "RUNNING_MIN_P":
		label = "실행 중 / RUNNING"
		style = okStyle
	case "VALIDATING_REUSE":
		label = "r6 재사용 검증 / VALIDATING REUSE"
		style = warnStyle
	case "RELEASING":
		label = "Router unload·VRAM 반환 확인 / RELEASING"
		style = warnStyle
	case "WAITING_FOR_JUDGMENT", "WAITING_FOR_FINAL_JUDGMENT":
		label = "최종 판정 대기 / WAITING FOR FINAL JUDGMENT"
		style = okStyle
	case "RESUME_REQUIRED":
		label = "자동 재개 대기 / RESUME REQUIRED"
		style = warnStyle
	case "FAILED_CLOSED", "RELEASE_FAILED":
		label = "안전 중단 / FAILED CLOSED"
		style = badStyle
	case "":
		label = "준비 중 / PREPARING"
		style = warnStyle
	default:
		style = warnStyle
	}
	manifest := ""
	if manifestStatus != "" {
		manifest = "  " + dimStyle.Render("artifact: "+manifestStatus)
	}
	return labelStyle.Render("상태 / STATE:") + " " + style.Render(label) + manifest
}

func formatETA(estimate ETAEstimate) string {
	switch estimate.Status {
	case "complete":
		return okStyle.Render("실행 완료 / EXECUTION COMPLETE")
	case "baseline":
		rangeText := ""
		if estimate.LowRemaining > 0 || estimate.HighRemaining > 0 {
			rangeText = fmt.Sprintf("  |  범위 약 %s~%s", formatDuration(estimate.LowRemaining), formatDuration(estimate.HighRemaining))
		}
		return strings.Join([]string{
			labelStyle.Render("초기 ETA / INITIAL ETA:") + " r6 실측 투영",
			fmt.Sprintf("%s  약 %s  |  %s%s", labelStyle.Render("예상 종료 / ETA:"), formatDuration(estimate.Remaining), estimate.FinishAt.In(time.Local).Format("01-02 15:04 MST"), dimStyle.Render(rangeText)),
		}, "\n")
	case "stalled":
		return warnStyle.Render("최근 progress 갱신이 90초를 넘었습니다. ETA를 일시 보류합니다 / No fresh completion for 90s; ETA paused.")
	case "stable", "measuring":
		confidence := "측정 중 / measuring"
		if estimate.Status == "stable" {
			confidence = "안정 / stable"
		}
		activeETA := estimate.Remaining
		if estimate.ActiveRemaining > 0 {
			activeETA = estimate.ActiveRemaining
		}
		backoff := ""
		if estimate.ExpectedBackoff > 0 {
			backoff = "  |  " + dimStyle.Render("예상 retry 대기 "+formatDuration(estimate.ExpectedBackoff))
		}
		return strings.Join([]string{
			fmt.Sprintf("%s  %.1f slots/min  |  %d개 최근 완료 표본, %s", labelStyle.Render("속도 / RATE:"), estimate.RatePerMinute, estimate.SampleCount, formatDuration(estimate.Observation)),
			fmt.Sprintf("%s  활성 약 %s%s", labelStyle.Render("활성 ETA / ACTIVE ETA:"), formatDuration(activeETA), backoff),
			fmt.Sprintf("%s  약 %s  |  %s  |  %s", labelStyle.Render("예상 종료 / ETA:"), formatDuration(estimate.Remaining), estimate.FinishAt.In(time.Local).Format("01-02 15:04 MST"), dimStyle.Render("recent/overall blend · "+confidence)),
		}, "\n")
	default:
		return dimStyle.Render("완료 표본을 수집 중입니다 / Collecting enough completion samples for ETA")
	}
}

func activityLine(snapshot Snapshot) string {
	if snapshot.ActiveSeconds <= 0 && snapshot.BackoffSeconds <= 0 {
		return dimStyle.Render("활성 시간 / ACTIVE: 아직 측정 중")
	}
	return fmt.Sprintf(
		"%s  %s  |  %s %s",
		labelStyle.Render("활성 시간 / ACTIVE:"),
		formatDuration(time.Duration(snapshot.ActiveSeconds*float64(time.Second))),
		labelStyle.Render("재시도 대기 / BACKOFF:"),
		formatDuration(time.Duration(snapshot.BackoffSeconds*float64(time.Second))),
	)
}

func lastCompletionLine(estimate ETAEstimate) string {
	if estimate.RecentComplete.IsZero() {
		return dimStyle.Render("마지막 완료 / LAST COMPLETE: 아직 없음")
	}
	return dimStyle.Render("마지막 완료 / LAST COMPLETE: ") + valueStyle.Render(estimate.RecentComplete.In(time.Local).Format("01-02 15:04:05 MST"))
}

func supervisorLogPath(runRoot string) string {
	managedRoot := managedRunRoot(runRoot)
	return filepath.Join(filepath.Dir(managedRoot), "supervisor-logs", filepath.Base(managedRoot)+".log")
}

func samplerLine(snapshot Snapshot) string {
	parts := []string{
		fmt.Sprintf("t=%.2f", snapshot.CurrentSampler.Temperature),
		fmt.Sprintf("top-p=%.2f", snapshot.CurrentSampler.TopP),
		fmt.Sprintf("top-k=%d", snapshot.CurrentSampler.TopK),
		fmt.Sprintf("min-p=%.2f", snapshot.CurrentSampler.MinP),
	}
	caseText := ""
	if snapshot.CaseCount > 0 && snapshot.CurrentCase > 0 {
		caseText = fmt.Sprintf("  |  seed %d · case %d/%d", snapshot.CurrentSeed, snapshot.CurrentCase, snapshot.CaseCount)
	}
	attempts := snapshot.Attempts
	return strings.Join([]string{
		labelStyle.Render("현재 sampler / CURRENT SAMPLER:") + " " + valueStyle.Render(strings.Join(parts, "  ")) + dimStyle.Render(caseText),
		fmt.Sprintf("%s  valid %s  |  retry %s  |  timeout %s  |  indeterminate %s", labelStyle.Render("응답 / RESPONSES:"), formatCount(attempts.Valid), formatCount(attempts.Retry), formatCount(attempts.Timeout), formatCount(attempts.Indeterminate)),
	}, "\n")
}

func freshnessLine(updatedAt, now time.Time) string {
	if updatedAt.IsZero() {
		return dimStyle.Render("파일 갱신 / FILE: 아직 수신하지 못함 / not received yet")
	}
	age := maxDuration(now.Sub(updatedAt), 0)
	return dimStyle.Render("파일 갱신 / FILE: ") + valueStyle.Render(formatDuration(age)+" 전 / ago") + dimStyle.Render("  ·  progress.json + completion index (read-only)")
}

func formatGPU(stats []GPUStat, readErr error, disabled bool) string {
	if disabled {
		return dimStyle.Render("GPU 조회 비활성 / GPU telemetry disabled")
	}
	if readErr != nil {
		return dimStyle.Render("NVIDIA telemetry unavailable; sampler execution is not affected.")
	}
	if len(stats) == 0 {
		return dimStyle.Render("GPU 텔레메트리 갱신 중 / Refreshing GPU telemetry")
	}
	lines := make([]string, 0, len(stats)*2)
	for _, stat := range stats {
		lines = append(lines, fmt.Sprintf(
			"GPU %s  %s",
			stat.Index,
			shortenMiddle(stat.Name, 48),
		))
		lines = append(lines, fmt.Sprintf(
			"  VRAM %d / %d MiB  |  util %d%%  |  %d°C",
			stat.MemoryUsed,
			stat.MemoryTotal,
			stat.Utilization,
			stat.Temperature,
		))
	}
	return valueStyle.Render(strings.Join(lines, "\n"))
}

func footerLine() string {
	return dimStyle.Render("q / Esc / Ctrl+C: monitor 창만 종료합니다. runner·Docker·GPU 작업은 계속됩니다.")
}

func formatDuration(duration time.Duration) string {
	if duration < time.Second {
		return "0s"
	}
	duration = duration.Round(time.Second)
	hours := duration / time.Hour
	minutes := (duration % time.Hour) / time.Minute
	seconds := (duration % time.Minute) / time.Second
	if hours > 0 {
		return fmt.Sprintf("%dh %dm", hours, minutes)
	}
	if minutes > 0 {
		return fmt.Sprintf("%dm %ds", minutes, seconds)
	}
	return fmt.Sprintf("%ds", seconds)
}

func formatCount(value int) string {
	text := fmt.Sprintf("%d", value)
	for index := len(text) - 3; index > 0; index -= 3 {
		text = text[:index] + "," + text[index:]
	}
	return text
}

func shortenMiddle(value string, limit int) string {
	if limit <= 3 || len(value) <= limit {
		return value
	}
	left := (limit - 1) / 2
	right := limit - 1 - left
	return value[:left] + "…" + value[len(value)-right:]
}

func maxDuration(left, right time.Duration) time.Duration {
	if left > right {
		return left
	}
	return right
}
