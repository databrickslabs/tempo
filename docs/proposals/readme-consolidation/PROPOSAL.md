# README Consolidation Proposal

## Status
📋 Proposed

## Created
October 2025

## Summary
Consolidate the duplicate README.md files (root and python/) into a single comprehensive README at the repository root.

---

## Problem

The Tempo repository currently maintains two README.md files:

1. **Root README.md** (24 lines)
   - Minimal content with basic project description
   - Links to documentation
   - Badges for build status, coverage, downloads
   - Last modified: September 2024

2. **python/README.md** (276 lines)
   - Comprehensive documentation with detailed examples
   - Quickstart guides for all major features
   - Installation instructions
   - Code examples for:
     - TSDF object creation
     - Resampling and visualization
     - AS OF joins
     - Moving averages (EMA, SMA)
     - Fourier transforms
     - Interpolation
     - Grouped statistics
   - Project setup and build instructions
   - Last modified: September 2024

### Issues with Current State

1. **Duplication**: Two READMEs have overlapping content but serve different purposes
2. **Confusion**: Users may not know which README is authoritative
3. **Maintenance burden**: Updates need to be synchronized across both files
4. **Discovery**: python/README.md is hidden from GitHub's main repository view
5. **Outdated content**: Root README lacks the comprehensive examples users need

Both files were created in the initial commit (July 14, 2020) and have existed side-by-side since then.

---

## Proposed Solution

### Option 1: Replace Root README with Enhanced Version (Recommended)

Consolidate both READMEs into a single, comprehensive README.md at the repository root.

**Structure:**
```markdown
# tempo - Time Series Utilities for Data Teams Using Databricks

[Logo]

## Project Description
[Enhanced description combining both versions]

[Badges from root README]

## [Tempo Project Documentation](link)

## Installation

### Using pip
- In Databricks notebooks
- Local installation

## Quick Start

[All examples from python/README.md]:
- TSDF object creation
- Resampling and visualization
- AS OF joins
- Skew-optimized joins
- Exponential moving average
- Simple moving average
- Fourier transform
- Interpolation
- Grouped statistics

## Project Support
[Support disclaimer]

## Contributing

### Development Setup
[Setup instructions from python/README.md]

### Building the Project
[Build instructions]

### Running Tests
[If applicable]

## License
[If applicable]
```

**Actions:**
1. Create new comprehensive README.md at root by:
   - Starting with python/README.md content
   - Adding badges from root README
   - Reorganizing sections for better flow
   - Updating any outdated information
2. Delete python/README.md
3. Update any references to python/README.md in documentation or CI/CD

### Option 2: Keep Minimal Root, Link to Comprehensive Version

Keep a minimal root README that links to python/README.md for details.

**Not recommended because:**
- GitHub users expect comprehensive README at root
- Adds unnecessary navigation step
- python/README.md is not prominently displayed

---

## Benefits

1. **Single source of truth**: One authoritative README
2. **Better discoverability**: Comprehensive docs visible on GitHub main page
3. **Reduced maintenance**: Only one file to update
4. **Improved user experience**: New users get all information immediately
5. **Professional appearance**: Standard repository structure

---

## Implementation Plan

### Phase 1: Content Consolidation
1. Review both READMEs for unique content
2. Identify badges, links, and metadata from root README
3. Identify comprehensive examples from python/README.md
4. Draft new consolidated README structure

### Phase 2: Create New README
1. Combine content following the proposed structure
2. Update any outdated information
3. Ensure all links work correctly
4. Verify code examples are current

### Phase 3: Cleanup
1. Back up python/README.md content (git history preserves it)
2. Delete python/README.md
3. Update any documentation references
4. Update CI/CD if it references python/README.md

### Phase 4: Validation
1. Review on GitHub preview
2. Verify all links work
3. Ensure badges display correctly
4. Get team approval

---

## Migration Checklist

- [ ] Audit both READMEs for unique content
- [ ] Draft consolidated README structure
- [ ] Create new README.md at root
- [ ] Verify all badges work
- [ ] Verify all links work
- [ ] Test code examples (optional but recommended)
- [ ] Search codebase for references to python/README.md
- [ ] Delete python/README.md
- [ ] Update docs/proposals/README.md to mark this as complete
- [ ] Commit changes

---

## Risks and Mitigation

### Risk 1: Breaking Links
**Impact**: External links to python/README.md may break
**Mitigation**:
- Search for external references before deletion
- GitHub should redirect automatically for most cases
- Document the change in CHANGELOG if needed

### Risk 2: Loss of Content
**Impact**: Important information from python/README.md could be lost
**Mitigation**:
- Comprehensive content audit before consolidation
- Git history preserves all content
- Review process before deletion

### Risk 3: CI/CD Dependencies
**Impact**: Build scripts might reference python/README.md
**Mitigation**:
- Search codebase for file references
- Update any build/deploy scripts

---

## Future Considerations

After consolidation, consider:

1. **Keep README focused**: Move detailed API docs to sphinx documentation
2. **Add more badges**: Code quality, license, latest release
3. **Add contributing guide**: Separate CONTRIBUTING.md for development workflow
4. **Update regularly**: Keep examples current with latest features
5. **Add changelog section**: Link to CHANGELOG.md for release notes

---

## Related Work

- CHANGELOG.md recently created at root
- Documentation reorganization (docs/proposals/ structure)
- Ongoing code quality improvements

---

## References

- Root README.md: 24 lines, basic project info
- python/README.md: 276 lines, comprehensive examples
- Both created: July 14, 2020 (initial commit)
- GitHub README best practices: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes
