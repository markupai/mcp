# Changelog

## [2.0.0] - 2024-12-19

### BREAKING CHANGES
- Updated to support new Markup AI API response structure
- Response format completely changed from flat to nested structure
- Workflow info now in `workflow` object with `id`, `status`, `api_version`
- Scores moved to `original.initial_scores`/`original.final_scores` 
- Issues now under `original.issues` with `position` object instead of `char_index`
- Rewritten text now at `rewrite.output.merged_text`

### Changed
- Updated tone enum values to: `academic`, `confident`, `conversational`, `empathetic`, `engaging`, `friendly`, `professional`, `technical`
- Default tone changed from `formal` to `professional`
- Removed unused interfaces and code
- Added ESLint configuration for code quality

### Fixed
- Fixed API 422 errors due to incorrect tone values
- Improved type safety by removing `any` types

## [1.0.0] - Initial Release
- Initial implementation of Markup AI MCP server
- Support for rewrite, check, and suggestions endpoints
- Async workflow polling support