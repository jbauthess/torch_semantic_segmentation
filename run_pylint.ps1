# run-pylint.ps1
# The CI linter step will fail only for pylint reporting errors or failures
# The console will report everything (so we can use those information to improve the code) 
try {
    # Run pylint and capture the exit code
    poetry run pylint .
    $pylintExitCode = $LASTEXITCODE
}
catch {
    $pylintExitCode = $_.ExitCode
}

# Check for errors or fatal issues (exit codes 1 or 2)
if ($pylintExitCode -band 3) {
    Write-Host "================================================================================="
    Write-Host "Pylint found errors or fatal issues, failing the build."
    Write-Host "================================================================================="
    exit 1
} else {
    Write-Host "================================================================================="
    Write-Host "Pylint found only warnings, refactors, or conventions. Not failing the build."
    Write-Host "================================================================================="
    exit 0
}