# DiffRhythm2.Net

C# port (.NET 8 + TorchSharp) of [DiffRhythm2](https://github.com/ASLP-lab/DiffRhythm2), the lyrics-and-style to song model.
This site holds the guides of the repository and the API reference, generated from the XML documentation comments in the source.

## Where to start

| You want to | Look at |
| --- | --- |
| Make a song from your own code | <xref:DiffRhythm2.Pipeline.SongGenerator> (a <xref:DiffRhythm2.Pipeline.SongRequest> in, audio out), and the [song pipeline guide](song-pipeline.md) |
| Make a song from a terminal | the `diffrhythm2` executable, in the [song pipeline guide](song-pipeline.md) |
| Run it as an HTTP service | the [HTTP service guide](http-service.md) (same API as the orchestrator's Python engine, with Swagger UI and `curl` examples) |
| Get the model files | the [weights tools guide](weights-tools.md) (download, licenses, conversion) |
| Recognize or compare songs | the [MuQ-MuLan audio guide](mulan-audio.md) (`embed` and `classify`) |
| Understand the model | the [diffusion transformer and sampler](dit-sampler.md), the [BigVGAN decoder](bigvgan-decoder.md), [lyrics to tokens](english-g2p.md) and [style text](mulan-text.md) |

Setup, requirements and the test commands are in the repository's README.
