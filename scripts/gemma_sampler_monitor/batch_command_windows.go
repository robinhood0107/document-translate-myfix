//go:build windows

package main

import (
	"os/exec"
	"syscall"
)

// newBatchCommand supplies CMD with its own unescaped command line. Go's normal
// Windows argument escaping is correct for CreateProcess targets, but CMD treats
// an escaped quote in a batch path as a literal backslash and cannot find the BAT.
func newBatchCommand(batchPath string, arguments ...string) *exec.Cmd {
	command := exec.Command("cmd.exe")
	command.SysProcAttr = &syscall.SysProcAttr{CmdLine: batchCommandLine(batchPath, arguments...)}
	return command
}
