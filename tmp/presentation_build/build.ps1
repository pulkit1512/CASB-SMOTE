$ErrorActionPreference='Stop'
$root='D:\SEM_5\ML_PROJECT'
$out=Join-Path $root 'tmp\presentation_build'
$spec=Get-Content (Join-Path $out 'slides.json') -Raw -Encoding UTF8|ConvertFrom-Json
New-Item -ItemType Directory -Force (Join-Path $out 'renders')|Out-Null
function RGB($hex) { [Convert]::ToInt32($hex.Substring(0,2),16)+256*[Convert]::ToInt32($hex.Substring(2,2),16)+65536*[Convert]::ToInt32($hex.Substring(4,2),16) }
$plum=RGB '39243D'; $ivory=RGB 'F6F2E9'; $ink=RGB '312C34'; $muted=RGB '706572'; $teal=RGB '217A70'; $clay=RGB 'B76243'; $white=RGB 'FFFFFF'
function Text($sl,$text,$x,$y,$w,$h,$size=22,$color=$ink,$bold=$false) {
 $sh=$sl.Shapes.AddTextbox(1,$x,$y,$w,$h)
 $tf=$sh.TextFrame; $tf.MarginLeft=0; $tf.MarginRight=0; $tf.MarginTop=0; $tf.MarginBottom=0; $tf.WordWrap=-1; $tf.AutoSize=0
 $tf.TextRange.Text=[string]$text; $tf.TextRange.Font.Name='Calibri'; $tf.TextRange.Font.Size=$size; $tf.TextRange.Font.Color.RGB=$color
 if($bold){$tf.TextRange.Font.Bold=-1}
 return $sh
}
function Line($sl,$x,$y,$x2,$y2,$color=$plum) {$sh=$sl.Shapes.AddLine($x,$y,$x2,$y2); $sh.Line.ForeColor.RGB=$color; $sh.Line.Weight=1.2; return $sh}
function Picture($sl,$path,$x,$y,$w,$h) {
 $im=[System.Drawing.Image]::FromFile($path); $ratio=[Math]::Min($w/$im.Width,$h/$im.Height); $pw=$im.Width*$ratio; $ph=$im.Height*$ratio; $im.Dispose()
 $null=$sl.Shapes.AddPicture($path,0,-1,$x+($w-$pw)/2,$y+($h-$ph)/2,$pw,$ph)
}
Add-Type -AssemblyName System.Drawing
$app=New-Object -ComObject PowerPoint.Application
try {
 $deck=$app.Presentations.Add(0); $deck.PageSetup.SlideWidth=960; $deck.PageSetup.SlideHeight=540
 $i=0
 foreach($s in $spec) {
  $i++; $sl=$deck.Slides.Add($i,12); $sl.FollowMasterBackground=0; $sl.Background.Fill.Solid(); $sl.Background.Fill.ForeColor.RGB=$ivory
  $dark=$s.kind -in @('cover','closing')
  if($dark){$sl.Background.Fill.ForeColor.RGB=$plum}
  $fc=$ink; if($dark){$fc=$ivory}
  $null=Text $sl $s.section.ToUpperInvariant() 42 22 850 22 11 $(if($dark){$ivory}else{$muted})
  $null=Text $sl ('{0:D2}' -f $i) 885 510 34 18 11 $fc
  if(-not $dark){$null=Text $sl $s.title 42 58 876 68 30 $plum $true; $null=Line $sl 42 129 918 129 (RGB 'D9D0D8')}
  switch($s.kind) {
   'cover' {
    $null=Text $sl $s.title 55 145 850 110 78 $ivory $true
    $null=Text $sl $s.subtitle 58 270 800 80 30 $ivory
    $null=Text $sl $s.authors 58 401 840 35 21 $ivory
    $null=Text $sl $s.affiliation 58 448 840 57 18 $ivory
   }
   'closing' {
    $null=Text $sl $s.title 55 155 850 110 78 $ivory $true
    $null=Text $sl $s.subtitle 58 275 830 100 29 $ivory
    $null=Text $sl $s.take 58 425 830 60 20 $ivory
   }
   'bullets' {
    $j=0; foreach($b in $s.bullets){$null=Text $sl ('{0:D2}' -f ($j+1)) 45 (161+$j*69) 45 40 16 $teal $true; $null=Text $sl $b 108 (156+$j*69) 795 60 25 $ink; $j++}
   }
   'table' {
    $nr=$s.rows.Count+1; $nc=$s.headers.Count
    $height=[Math]::Min(310,50*$nr); if($s.table_height){$height=$s.table_height}; $sh=$sl.Shapes.AddTable($nr,$nc,42,150,876,$height); $tb=$sh.Table
    if($s.widths){for($c=1;$c -le $nc;$c++){$tb.Columns.Item($c).Width=876*$s.widths[$c-1]}}
    for($r=1;$r -le $nr;$r++){for($c=1;$c -le $nc;$c++){
     $cell=$tb.Cell($r,$c).Shape; $tf=$cell.TextFrame; $tf.MarginLeft=9; $tf.MarginRight=9; $tf.MarginTop=7; $tf.MarginBottom=5; $tf.VerticalAnchor=3
     $v=$s.headers[$c-1]; if($r -gt 1){$v=$s.rows[$r-2][$c-1]}
     $tf.TextRange.Text=[string]$v; $tf.TextRange.Font.Name='Calibri'; $tf.TextRange.Font.Size=18; if($s.small){$tf.TextRange.Font.Size=16}
     $tf.TextRange.Font.Color.RGB=$ink; $cell.Fill.Solid(); $cell.Fill.ForeColor.RGB=$white
     if($r % 2 -eq 0){$cell.Fill.ForeColor.RGB=RGB 'EEE8ED'}
     if($r -eq 1){$cell.Fill.ForeColor.RGB=$plum; $tf.TextRange.Font.Color.RGB=$white; $tf.TextRange.Font.Bold=-1}
     foreach($bc in $s.bold_cells){if($r -eq $bc[0] -and $c -eq $bc[1]){$tf.TextRange.Font.Bold=-1; $tf.TextRange.Font.Color.RGB=$teal}}
     foreach($edge in 1..4){$bd=$tb.Cell($r,$c).Borders.Item($edge); $bd.ForeColor.RGB=$ivory; $bd.Weight=1}
    }}
   }
   'references' {
    for($j=0;$j -lt $s.entries.Count;$j++){
     $x=42+($j%2)*450; $y=145+[Math]::Floor($j/2)*109; $e=$s.entries[$j]
     $null=Text $sl $e[0] $x $y 426 24 17 $teal $true
     $null=Text $sl $e[1] $x ($y+25) 426 57 15 $ink
     $pub=($e[2] -split "`n")[0]
     $null=Text $sl $pub $x ($y+83) 426 23 13 $muted
    }
   }
   'plot' {Picture $sl $s.image 42 140 876 326}
   'flow' {
    for($j=0;$j -lt $s.steps.Count;$j++){
     $x=42+$j*226; $box=$sl.Shapes.AddShape(1,$x,205,198,108); $box.Fill.ForeColor.RGB=$plum; $box.Line.Visible=0
     $null=Text $sl ('0'+($j+1)) $x 166 198 26 16 $teal $true
     $null=Text $sl $s.steps[$j] ($x+13) 225 172 72 23 $white $true
     $null=Text $sl $s.details[$j] $x 331 198 93 19 $muted
     if($j -lt 3){$arr=Line $sl ($x+201) 260 ($x+222) 260 $teal; $arr.Line.EndArrowheadStyle=3; $arr.Line.Weight=2}
    }
   }
   {$_ -in @('problem','numbers','contrast')} {
    $n=$s.values.Count; $cw=876/$n
    for($j=0;$j -lt $n;$j++){
     $x=42+$cw*$j; $null=Text $sl $s.values[$j] $x 157 ($cw-20) 102 82 $(if($j -eq 0){$teal}else{$clay}) $true
     $null=Text $sl $s.labels[$j] $x 265 ($cw-26) 104 23 $ink
    }
    if($s.bullets){$j=0; foreach($b in $s.bullets){$null=Text $sl $b 46 (373+$j*26) 866 28 18 $muted; $j++}}
   }
  }
  if($s.insight){$null=Text $sl $s.insight 42 419 876 54 20 $teal}
  if($s.take -and -not $dark){$ts=17; if($s.take_size){$ts=$s.take_size}; $null=Text $sl $s.take 42 480 829 52 $ts $plum $true}
  $sl.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text=$s.notes+"`r`n`r`nEvidence: "+$s.source
 }
 $candidate=Join-Path $out 'CASB_Research_Presentation.pptx'
 $deck.SaveAs($candidate,24)
 foreach($sl in $deck.Slides){$sl.Export((Join-Path $out ('renders\slide_{0:D2}.png' -f $sl.SlideIndex)),'PNG',1600,900)}
 $deck.Close()
 "Created $i slides: $candidate"
} finally {if($app){$app.Quit(); [Runtime.InteropServices.Marshal]::ReleaseComObject($app)|Out-Null}}
