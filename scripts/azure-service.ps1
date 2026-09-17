# Inicia o detiene el App Service de Azure para ahorrar crédito de
# Azure for Students (se cobra por hora que el plan esté "En ejecución",
# sin importar si recibe tráfico o no).
#
# Uso:
#   .\scripts\azure-service.ps1 start   # antes de una demo/prueba
#   .\scripts\azure-service.ps1 stop    # al terminar de usarlo
#   .\scripts\azure-service.ps1 status  # ver el estado actual
#
# Requiere Azure CLI instalado y sesión iniciada (az login), una sola vez.

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("start", "stop", "status")]
    [string]$Action
)

$ResourceGroup = "rg-visionnav"
$AppName = "visionnav-api"

switch ($Action) {
    "start" {
        Write-Host "Iniciando $AppName..." -ForegroundColor Cyan
        az webapp start --resource-group $ResourceGroup --name $AppName | Out-Null
        Write-Host "Iniciado. El contenedor tarda ~30-60s en estar listo para recibir tráfico." -ForegroundColor Green
        Write-Host "URL: https://visionnav-api-c5g0gxamh2bug7be.canadacentral-01.azurewebsites.net" -ForegroundColor Gray
    }
    "stop" {
        Write-Host "Deteniendo $AppName (esto corta el cobro por hora)..." -ForegroundColor Cyan
        az webapp stop --resource-group $ResourceGroup --name $AppName | Out-Null
        Write-Host "Detenido." -ForegroundColor Green
    }
    "status" {
        $state = az webapp show --resource-group $ResourceGroup --name $AppName --query "state" -o tsv
        Write-Host "Estado actual: $state" -ForegroundColor Cyan
    }
}
