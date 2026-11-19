$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$part = Join-Path $taskRoot 'experiments\astra\sq3_iterative_review_20261026_v1\part2'

function Read-Json([string]$Path) {
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -AsHashtable
}

function File-Sha([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Check-Hashes($Mapping) {
    foreach ($name in $Mapping.Keys) {
        $actual = File-Sha (Join-Path $taskRoot $name)
        if ($actual -cne $Mapping[$name]) { throw "Changed bound file: $name" }
    }
    return $Mapping.Count
}

$done = Read-Json (Join-Path $part 'complete.json')
if ($done.status -ne 'iterative_part2_complete_reviewed' -or $done.whole_study_closed) {
    throw 'Wrong Part 2 completion state'
}
$checked = Check-Hashes $done.components
$reader = Read-Json (Join-Path $part 'reader_v3\complete.json')
$checked += Check-Hashes $reader.input_bindings
$checked += Check-Hashes $reader.artifacts
$numbers = Read-Json (Join-Path $part 'errors_v1\independent_verification.json')
if ($numbers.status -ne 'independent_four_round_numerical_verification_pass' -or
    $numbers.exact_fraction_points -ne 440 -or $numbers.bootstrap_scalars_recomputed -ne 4400000 -or
    $numbers.maximum_bootstrap_difference -ne 0) { throw 'Numerical proof differs' }
$contract = Read-Json (Join-Path $part 'analysis_contract_v3.json')
$checked += Check-Hashes $contract.source_hashes
$publication = Read-Json (Join-Path $part 'publication_v2\complete.json')
$checked += Check-Hashes $publication.outputs
foreach ($relative in $publication.previous_hashes.Keys) {
    $original = Join-Path $taskRoot $relative
    $backup = Join-Path $part ('publication_v2\backups\' + (Split-Path -Leaf $relative))
    if ((File-Sha $backup) -cne $publication.previous_hashes[$relative]) { throw 'Backup changed' }
    if ($relative -match 'v91\.md$|v7\.md$') {
        if ((File-Sha $original) -cne (File-Sha $backup)) { throw 'Old roadmap or checklist changed' }
    } else {
        $oldBytes = [IO.File]::ReadAllBytes($backup)
        $newBytes = [IO.File]::ReadAllBytes($original)
        $offset = [Array]::IndexOf($oldBytes, [byte]10) + 1
        $length = $oldBytes.Length - $offset
        $oldTail = [Convert]::ToBase64String($oldBytes, $offset, $length)
        $newTail = [Convert]::ToBase64String($newBytes, $newBytes.Length - $length, $length)
        if ($oldTail -cne $newTail) { throw 'Historical log bytes changed' }
    }
}
$roadmapPath = Join-Path $taskRoot 'Thesis_Log\astra_result\astra_remaining_work_before_writing_20261026_v92.md'
$roadmap = Get-Content -LiteralPath $roadmapPath -Raw
$experimental = (($roadmap -split '## The nine experimental work items', 2)[1] -split '\r?\n## ', 2)[0]
$completeItems = [regex]::Matches($experimental, '(?m)^- \[x\]').Count
$openItems = [regex]::Matches($experimental, '(?m)^- \[ \]').Count
$inventory = (($roadmap -split '## Full accepted inventory retained from the previous roadmap', 2)[1] -split '\r?\n## ', 2)[0]
$inventoryItems = @($inventory -split '\r?\n' | Where-Object { $_.StartsWith('| ') }).Count - 1
if ($completeItems -ne 4 -or $openItems -ne 5 -or $inventoryItems -ne 16) { throw 'Remaining-work inventory differs' }
$reportPath = Join-Path $taskRoot $done.report
$links = 0
foreach ($name in @($done.report,
    'Thesis_Log/astra_result/astra_sq3_iterative_review_progress_20261026_v2.md',
    'Thesis_Log/astra_result/astra_sq3_completion_checklist_20261026_v8.md')) {
    $path = Join-Path $taskRoot $name
    $text = Get-Content -LiteralPath $path -Raw
    foreach ($match in [regex]::Matches($text, '\]\(([^)]+)\)')) {
        $target = ($match.Groups[1].Value -split '#', 2)[0]
        if ($target -notmatch '://') {
            if (-not (Test-Path -LiteralPath (Join-Path (Split-Path -Parent $path) $target))) { throw "Broken link: $target" }
        }
        $links++
    }
}
$evidence = [ordered]@{
    status = 'independent_powershell_publication_readback_pass'
    created_utc = [DateTime]::UtcNow.ToString('o')
    part2_complete_sha256 = File-Sha (Join-Path $part 'complete.json')
    script_sha256 = File-Sha $PSCommandPath
    verified_hash_bindings = $checked
    verified_links = $links
    historical_log_bytes_preserved = $true
    original_roadmap_and_checklist_preserved = $true
    complete_experimental_items = $completeItems
    open_experimental_items = $openItems
    accepted_inventory_entries = $inventoryItems
    bootstrap_scalars_independently_verified = $numbers.bootstrap_scalars_recomputed
    new_Qwen_calls = 0
    new_fits = 0
}
$destination = Join-Path $part 'publication_v2\readback.json'
if (Test-Path -LiteralPath $destination) { throw 'Preserve previous readback evidence' }
$json = $evidence | ConvertTo-Json -Depth 6
[IO.File]::WriteAllText($destination, $json + "`n", [Text.UTF8Encoding]::new($false))
$json
