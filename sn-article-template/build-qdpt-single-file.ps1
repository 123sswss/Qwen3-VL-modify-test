param(
    [string]$Source = '.\qdpt-sn-source.tex',
    [string]$Output = '.\qdpt-sn.tex'
)

$sourcePath = (Resolve-Path -LiteralPath $Source).Path
$sourceDir = Split-Path -Parent $sourcePath
$lines = Get-Content -LiteralPath $sourcePath -Encoding UTF8
$expanded = foreach ($line in $lines) {
    if ($line -match '^\s*\\input\{([^}]+)\}\s*$') {
        $inputPath = Join-Path $sourceDir ($Matches[1] + '.tex')
        Get-Content -LiteralPath $inputPath -Encoding UTF8
    }
    else {
        $line
    }
}

Set-Content -LiteralPath $Output -Value $expanded -Encoding UTF8
