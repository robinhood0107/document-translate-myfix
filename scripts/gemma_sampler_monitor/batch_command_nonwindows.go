//go:build !windows

package main

import "os/exec"

// The launcher ships for Windows. This fallback keeps static Go tests portable
// when they run from the WSL checkout.
func newBatchCommand(batchPath string, arguments ...string) *exec.Cmd {
	return exec.Command("cmd.exe", "/d", "/c", batchCommandTail(batchPath, arguments...))
}
