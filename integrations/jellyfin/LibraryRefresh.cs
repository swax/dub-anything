using System.Text.Json;
using MediaBrowser.Controller;
using MediaBrowser.Controller.Library;
using MediaBrowser.Controller.Plugins;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace Jellyfin.Plugin.DubAnything;

public sealed class ServiceRegistrator : IPluginServiceRegistrator
{
    public void RegisterServices(IServiceCollection services, IServerApplicationHost applicationHost)
        => services.AddHostedService<LibraryRefresh>();
}

// Publication does not require leaving a browser tab open.
public sealed class LibraryRefresh(ILibraryManager library, ILogger<LibraryRefresh> logger) : BackgroundService
{
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        using var timer = new PeriodicTimer(TimeSpan.FromSeconds(30));
        do
        {
            try
            {
                var plugin = Plugin.Instance;
                if (plugin is null || string.IsNullOrWhiteSpace(plugin.Configuration.ServiceToken))
                    continue;
                using var heartbeat = await WorkerClient.Send(HttpMethod.Post, "plugin", new
                {
                    version = typeof(Plugin).Assembly.GetName().Version!.ToString(),
                    server_version = typeof(IServerApplicationHost).Assembly.GetName().Version!.ToString()
                }, stoppingToken);
                using var response = await WorkerClient.Send(HttpMethod.Get, "jobs", null, stoppingToken);
                response.EnsureSuccessStatusCode();
                using var document = JsonDocument.Parse(await response.Content.ReadAsStringAsync(stoppingToken));
                var known = plugin.Configuration.RefreshedJobIds.ToHashSet();
                var published = document.RootElement.EnumerateArray()
                    .Where(job => job.GetProperty("state").GetString() == "completed")
                    .Select(job => job.GetProperty("id").GetString()!)
                    .Where(id => !known.Contains(id)).ToArray();
                if (published.Length == 0)
                    continue;
                library.QueueLibraryScan();
                plugin.Configuration.RefreshedJobIds = known.Concat(published).ToArray();
                plugin.SaveConfiguration();
                logger.LogInformation("Queued library scan after {Count} published dubs", published.Length);
            }
            catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested) { break; }
            catch (Exception exception)
            {
                logger.LogDebug(exception, "Dub Anything worker is unavailable; will check again");
            }
        } while (await timer.WaitForNextTickAsync(stoppingToken));
    }
}
