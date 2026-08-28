[CmdletBinding()]
param()

$ErrorActionPreference = 'SilentlyContinue'
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

if ([Console]::IsOutputRedirected) { exit 0 }

$Source = @'
using System;
using System.Runtime.InteropServices;

public static class ComicConsoleFont {
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    public struct CONSOLE_FONT_INFOEX {
        public uint cbSize;
        public uint nFont;
        public short FontWidth;
        public short FontHeight;
        public int FontFamily;
        public int FontWeight;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)]
        public string FaceName;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr GetStdHandle(int nStdHandle);

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern bool SetCurrentConsoleFontEx(
        IntPtr consoleOutput,
        bool maximumWindow,
        ref CONSOLE_FONT_INFOEX consoleCurrentFontEx
    );

    public static bool Set(string faceName, short height) {
        var info = new CONSOLE_FONT_INFOEX();
        info.cbSize = (uint)Marshal.SizeOf(info);
        info.nFont = 0;
        info.FontWidth = 0;
        info.FontHeight = height;
        info.FontFamily = 54;
        info.FontWeight = 400;
        info.FaceName = faceName;
        return SetCurrentConsoleFontEx(GetStdHandle(-11), false, ref info);
    }
}
'@

if (-not ('ComicConsoleFont' -as [type])) {
    Add-Type -TypeDefinition $Source -Language CSharp | Out-Null
}
foreach ($Face in @('Cascadia Mono', 'Consolas')) {
    if ([ComicConsoleFont]::Set($Face, 18)) { break }
}
