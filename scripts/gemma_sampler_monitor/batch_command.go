package main

import "strings"

func batchCommandLine(batchPath string, arguments ...string) string {
	return "cmd.exe /d /c " + batchCommandTail(batchPath, arguments...)
}

func batchCommandTail(batchPath string, arguments ...string) string {
	parts := make([]string, 0, len(arguments)+2)
	parts = append(parts, "call", quoteBatchToken(batchPath))
	for _, argument := range arguments {
		parts = append(parts, quoteBatchToken(argument))
	}
	return strings.Join(parts, " ")
}

func quoteBatchToken(value string) string {
	return `"` + strings.ReplaceAll(value, `"`, `""`) + `"`
}
