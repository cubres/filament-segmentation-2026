# Publication boundary

This repository publishes an explicit file allowlist.
It contains original source, contract tests, package pins, prose, aggregate local results, a split manifest, and generated art.
The notebook has no executed outputs or competition-image attachments.

The release excludes:

- Competition images, labels, annotations, and predicted mask files.
- The downloaded host notebook and evaluator source.
- Native agent transcripts, private logs, credentials, and local absolute paths.
- Local dependency trees and environment folders.

Before publication, the release scan checks text for common credential formats and local user paths.
It checks notebook code and attachments separately.
This is a focused release review, not a guarantee that arbitrary content is secret-free.

Never delete, rename, replace, or make any notebook identity unreachable.
Keep the public notebook identity stable. Update existing files in place.
Keep prior versions and source evidence.
