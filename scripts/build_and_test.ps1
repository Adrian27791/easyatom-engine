# Build directo con clang++ del NDK (sin CMake, sin Ninja).
# Uso: pwsh ./scripts/build_and_test.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$cl = "C:/Users/DnR/AppData/Local/Android/SDK/ndk/29.0.14033849/toolchains/llvm/prebuilt/windows-x86_64/bin/clang++.exe"
if (-not (Test-Path $cl)) { throw "No se encontro clang++ en NDK 29: $cl" }

$flags = @(
    '-std=c++20',
    '-Wall', '-Wextra', '-Wpedantic', '-Wshadow',
    '-O2',
    '-Iinclude', '-Itests'
)
$sources = @(
    'tests/test_main.cpp',
    'tests/test_clifford.cpp',
    'tests/test_hilbert.cpp',
    'tests/test_ops.cpp',
    'tests/test_infogeo.cpp',
    'tests/test_topology.cpp',
    'tests/test_dynamics.cpp',
    'tests/test_laws.cpp'
)
$out = 'test_easyatom.exe'

Write-Host ">> Compilando..." -ForegroundColor Cyan
& $cl @flags @sources -o $out
if ($LASTEXITCODE -ne 0) { throw "Fallo de compilacion." }

Write-Host ">> Ejecutando tests..." -ForegroundColor Cyan
& ".\$out"
if ($LASTEXITCODE -ne 0) { throw "Tests fallaron." }
Write-Host ">> OK" -ForegroundColor Green
