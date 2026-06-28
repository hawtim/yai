param(
  [string]$HostAddress = "0.0.0.0",
  [int]$Port = 8765,
  [string]$ApiKey = "",
  [int]$MaxWorkers = 1,
  [int]$QueueSize = 100,
  [int]$CooldownSeconds = 60,
  [int]$CooldownMaxSeconds = 300,
  [double]$CooldownGpuTemp = 65,
  [double]$CooldownGpuUtilization = 20
)

if ($ApiKey) {
  $env:YAI_API_KEY = $ApiKey
}

Set-Location "D:\Dev\youtube-audio-intel"
yai serve --host $HostAddress --port $Port --max-workers $MaxWorkers --queue-size $QueueSize --cooldown-seconds $CooldownSeconds --cooldown-max-seconds $CooldownMaxSeconds --cooldown-gpu-temp $CooldownGpuTemp --cooldown-gpu-utilization $CooldownGpuUtilization
