using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using MediaBrowser.Common.Api;
using MediaBrowser.Controller.Entities;
using MediaBrowser.Controller.Library;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace Jellyfin.Plugin.DubAnything;

// Requests are authenticated by Jellyfin; only administrators can launch agents.
[ApiController]
[Route("DubAnything")]
[Authorize(Policy = Policies.RequiresElevation)]
public sealed class DubController(ILibraryManager library) : ControllerBase
{
    [HttpGet("Info")]
    public IActionResult Info() => Ok(new
    {
        version = typeof(Plugin).Assembly.GetName().Version?.ToString() ?? "unknown"
    });

    [HttpGet("Capabilities")]
    public Task<IActionResult> Capabilities(CancellationToken ct) => Forward(HttpMethod.Get, "capabilities", null, ct);

    [HttpGet("Updates")]
    public Task<IActionResult> Updates(CancellationToken ct) => Forward(HttpMethod.Get, "updates", null, ct);

    [HttpGet("Jobs")]
    public Task<IActionResult> Jobs(CancellationToken ct) => Forward(HttpMethod.Get, "jobs", null, ct);

    [HttpPost("Jobs")]
    public Task<IActionResult> Create([FromBody] DubRequest request, CancellationToken ct)
    {
        var item = library.GetItemById(request.ItemId);
        if (item is not Video || string.IsNullOrEmpty(item.Path) || !System.IO.File.Exists(item.Path))
            return Task.FromResult<IActionResult>(BadRequest(new { error = "Choose a local video from your Jellyfin library." }));
        var config = Plugin.Instance!.Configuration;
        var director = string.IsNullOrWhiteSpace(request.Director) ? config.DefaultDirector : request.Director;
        var payload = new
        {
            item_id = item.Id.ToString("N"),
            title = item.Name,
            path = item.Path,
            source_lang = request.SourceLanguage,
            target_lang = request.TargetLanguage,
            voices = request.Voices,
            director,
            model = director == "claude" ? config.ClaudeModel : config.CodexModel,
            review_before_publish = config.ReviewBeforePublish
        };
        return Forward(HttpMethod.Post, "jobs", payload, ct);
    }

    [HttpPost("Jobs/{id:guid}/{action}")]
    public Task<IActionResult> Action(Guid id, string action, CancellationToken ct)
    {
        if (action is not ("cancel" or "resume" or "publish"))
            return Task.FromResult<IActionResult>(NotFound());
        return Forward(HttpMethod.Post, $"jobs/{id:N}/{action}", new { }, ct);
    }

    [HttpGet("Jobs/{id:guid}/{kind}")]
    public Task<IActionResult> Details(Guid id, string kind, CancellationToken ct)
    {
        if (kind is not ("report" or "log"))
            return Task.FromResult<IActionResult>(NotFound());
        return Forward(HttpMethod.Get, $"jobs/{id:N}/{kind}", null, ct);
    }

    private async Task<IActionResult> Forward(HttpMethod method, string path, object? payload, CancellationToken ct)
    {
        try
        {
            using var response = await WorkerClient.Send(method, path, payload, ct);
            return new ContentResult
            {
                StatusCode = (int)response.StatusCode,
                ContentType = "application/json",
                Content = await response.Content.ReadAsStringAsync(ct)
            };
        }
        catch (Exception exception) when (exception is HttpRequestException or TaskCanceledException or InvalidOperationException)
        {
            return StatusCode(503, new { error = "Cannot reach the dubbing worker. Check its address, token, and service log." });
        }
    }
}

public sealed class DubRequest
{
    public Guid ItemId { get; set; }
    public string Director { get; set; } = "";
    public string SourceLanguage { get; set; } = "ja";
    public string TargetLanguage { get; set; } = "en";
    public string Voices { get; set; } = "actor";
}

internal static class WorkerClient
{
    private static readonly HttpClient Client = new(new SocketsHttpHandler { UseProxy = false })
    {
        Timeout = TimeSpan.FromSeconds(15)
    };

    public static async Task<HttpResponseMessage> Send(HttpMethod method, string path, object? payload, CancellationToken ct)
    {
        var config = Plugin.Instance!.Configuration;
        if (!Uri.TryCreate(config.ServiceUrl.TrimEnd('/') + "/", UriKind.Absolute, out var root)
            || root.Scheme != "http" || !root.IsLoopback || string.IsNullOrEmpty(config.ServiceToken))
            throw new InvalidOperationException("Configure a loopback worker URL and its service token.");
        using var request = new HttpRequestMessage(method, new Uri(root, path));
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", config.ServiceToken);
        if (payload is not null)
            request.Content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json");
        return await Client.SendAsync(request, ct);
    }
}
