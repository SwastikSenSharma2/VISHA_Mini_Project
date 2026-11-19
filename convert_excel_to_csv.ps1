$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false
$wb = $excel.Workbooks.Open('C:\Users\Swastik Sen Sharma\Downloads\VISHA-main\SBE-profiles.xlsx')
$wb.SaveAs('C:\Users\Swastik Sen Sharma\Downloads\VISHA-main\SBE-profiles.csv', 6)
$wb.Close($false)
$excel.Quit()
