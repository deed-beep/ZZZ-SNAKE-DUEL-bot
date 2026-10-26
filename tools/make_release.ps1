<#
.SYNOPSIS
    生成「蛇对蛇助手 2.5」发行包：按白名单收集文件、重算 文件校验.json、压成 zip。

.DESCRIPTION
    只收集运行所需与合规说明，自动排除 .git、.github、tools、docs、验收记录、
    README、CHANGELOG 等仓库专用内容。校验清单格式与 源码/release.py 一致：
    正斜杠相对路径 + 小写 SHA-256。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\tools\make_release.ps1 -Version 1.0.2

.EXAMPLE
    .\tools\make_release.ps1 -Version 1.0.3 -NoSource -Output D:\发布\SnakeAssistant25-1.0.3-win64.zip
#>
[CmdletBinding()]
param(
    [string]$Version = "1.0.2",
    [string]$Source,
    [string]$Output,
    [switch]$NoSource
)

$ErrorActionPreference = "Stop"

$root = if ($Source) { (Resolve-Path -LiteralPath $Source).Path } else { Split-Path -Parent $PSScriptRoot }
$packageName = "蛇对蛇助手2.5"
$stage = Join-Path $env:TEMP ("snake25_release_" + [guid]::NewGuid().ToString("N").Substring(0, 8))
$package = Join-Path $stage $packageName
$zipPath = if ($Output) { $Output } else { Join-Path ([Environment]::GetFolderPath("Desktop")) ("SnakeAssistant25-$Version-win64.zip") }

# ---- 白名单：运行必需 + 合规 + 用户文档 ----
$items = @(
    "SnakeAssistant25.exe",
    "_internal",
    "使用声明与免责声明.txt",
    "驱动安装包",
    "第三方许可",
    "使用说明.md",
    "策略与参数说明.md",
    "声明保护说明.md",
    "runtime_versions.json",
    "LICENSE"
)
$items += @(Get-ChildItem -LiteralPath $root -Filter "更新说明_*.md" -File | ForEach-Object { $_.Name })
if (-not $NoSource) { $items += "源码" }

foreach ($required in @("SnakeAssistant25.exe", "_internal", "使用声明与免责声明.txt")) {
    if (-not (Test-Path -LiteralPath (Join-Path $root $required))) { throw "缺少必需项：$required" }
}

Write-Host "源目录：$root"
Write-Host "输出：  $zipPath"

# ---- 收集 ----
New-Item -ItemType Directory -Path $package -Force | Out-Null
foreach ($name in $items) {
    $sourcePath = Join-Path $root $name
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        Write-Warning "跳过不存在的项：$name"
        continue
    }
    Copy-Item -LiteralPath $sourcePath -Destination $package -Recurse -Force
    Write-Host ("  已收集 {0}" -f $name)
}

# ---- 清理源码目录里的构建残留（只动 源码，避免误删运行库） ----
$sourceDir = Join-Path $package "源码"
if (Test-Path -LiteralPath $sourceDir) {
    Get-ChildItem -LiteralPath $sourceDir -Recurse -Directory -Force |
        Where-Object { $_.Name -in @("__pycache__", "dist", "build") } |
        ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }
    Get-ChildItem -LiteralPath $sourceDir -Recurse -File -Force |
        Where-Object { $_.Extension -in @(".pyc", ".spec") } |
        ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
}

# ---- 重算校验清单 ----
$hashes = [ordered]@{}
$files = Get-ChildItem -LiteralPath $package -Recurse -File -Force | Sort-Object { $_.FullName }
foreach ($file in $files) {
    $relative = $file.FullName.Substring($package.Length + 1).Replace("\", "/")
    if ($relative -eq "文件校验.json") { continue }
    $hashes[$relative] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
}
$json = $hashes | ConvertTo-Json -Depth 3
[System.IO.File]::WriteAllText((Join-Path $package "文件校验.json"), $json, (New-Object System.Text.UTF8Encoding($false)))

# ---- 打包 ----
if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
$zipDir = Split-Path -Parent $zipPath
if ($zipDir -and -not (Test-Path -LiteralPath $zipDir)) { New-Item -ItemType Directory -Path $zipDir -Force | Out-Null }
& tar -a -c -f $zipPath -C $stage $packageName
if ($LASTEXITCODE -ne 0) { throw "tar 打包失败，退出码 $LASTEXITCODE" }

# ---- 结果 ----
$sizeMb = [math]::Round((Get-Item -LiteralPath $zipPath).Length / 1MB, 1)
$fileCount = (Get-ChildItem -LiteralPath $package -Recurse -File -Force).Count
Write-Host ""
Write-Host "完成：$zipPath" -ForegroundColor Green
Write-Host ("包内文件 {0} 个，校验项 {1} 条，zip 大小 {2} MB" -f $fileCount, $hashes.Count, $sizeMb)
Write-Host "临时目录（可删）：$stage"
