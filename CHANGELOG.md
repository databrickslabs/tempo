# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.30] - 2025-08-28

### Added

- Enhanced Makefile functionality with environment checks for Python and Java setups
- New `formatBlack` command in Makefile for code formatting
- Support for more column types in interpolation methods
- `_cleanup_delta_warehouse` method for Delta test environment cleanup
- Dynamic Python environment management with pyenv integration
- Coverage report generation with parallel coverage file handling
- Artifact signing with Sigstore for PyPI releases
- Draft release creation in GitHub Actions workflow

### Changed

- Upgraded build system from Tox to Hatch
- Updated release process to not rely on git tags
- Migrated CI/CD workflows to use make commands instead of direct tox calls
- Updated Python version support (removed 3.8, added 3.11)
- Enhanced type checking with mypy improvements
- Improved error handling in coverage report generation
- Switched release workflow to use protected runner group (databrickslabs-protected-runner-group)

### Fixed

- Coverage report generation error in CI/CD environments
- Fixed handling of parallel coverage files (.coverage.*)
- Resolved unit test incompatibility with DBR 1.13
- Fixed bool conversions in tempo/intervals.py for consistent type handling
- Corrected column reference errors in interpolation tests
- Fixed mypy type checking issues with scipy

### Infrastructure

- Updated GitHub workflows to use Hatch instead of Tox
- Enhanced Makefile with better Python version management
- Added pre- and post-test cleanup for Delta warehouse directories
- Improved virtual environment handling and creation
- Enhanced release workflow with protected runner group for security
- Added automated artifact signing for Python distribution packages

## Detailed Changes

### Update release workflow: switch to protected runner group, add artifact signing, and draft releases (#436)

**Commit:** c8002f29  
**Author:** Lorin Dawson  
**Date:** 2025-09-10

- Switched release workflow to use protected runner group (databrickslabs-protected-runner-group) for enhanced security
- Added draft release creation using softprops/action-gh-release@v2 with wheel distribution files
- Integrated Sigstore artifact signing for Python packages (dbl_tempo*.whl and tempo*.whl)
- Enhanced release process with automated artifact signing and release management

### Fix coverage report generation error (#434)

**Commit:** 1c31f6a  
**Author:** Lorin Dawson  
**Date:** 2025-08-28

- Removed skip-install from coverage-report environment to ensure source files are accessible
- Added automatic handling of parallel coverage files in CI/CD environments
- Improved error messages when coverage data is missing
- Preserved test exit codes for proper CI/CD failure reporting

### Updating release process to not rely on tags

**Commit:** 7273c73  
**Author:** kwang-databricks  
**Date:** 2025-08-06

- Updated release process to remove dependency on git tags
- Manual version management changes
- Cleaned up get_version function to remove git references

### Added formatBlack in makefile, updated hatch files (#429)

**Commit:** 2476e8e  
**Author:** kwang-databricks  
**Date:** 2025-08-06

- Added formatBlack command to Makefile for consistent code formatting
- Updated hatch configuration files with respective commands
- Removed erroneous ignore file

### Allow more column types to be interpolated (#421)

**Commit:** 74c2b07  
**Author:** Brian Deacon  
**Date:** 2025-06-30

- Expanded interpolation functionality to support additional column types
- Fixed unit test compatibility issues with DBR 1.13
- Improved error messages for unsupported column types
- Enhanced test coverage for column interpolation methods

### Upgrading Tox to Hatch, and Updating respective commands in Makefile (#426)

**Commit:** 042c309  
**Author:** kwang-databricks  
**Date:** 2025-06-24

- Migrated from Tox to Hatch for improved build and environment management
- Updated all CI/CD workflows to use make commands
- Added comprehensive Makefile with environment management features
- Enhanced Python version support and management
- Improved type checking configuration with mypy
- Added version handling for hatch environment creation
- Updated GitHub workflow dependencies to use hatch instead of tox

### [Chore] CICD updates 02 (#425)

**Commit:** 8597388  
**Author:** kwang-databricks  
**Date:** 2025-06-13

- Enhanced Makefile with Python environment checks and management
- Added Delta warehouse cleanup methods for test environments
- Improved virtual environment handling and creation
- Updated .gitignore to include .claude directory
- Fixed bool type conversions throughout codebase
- Added setup-python-versions and setup-all-python-versions targets

### Created prelim makefile with tox commands, updated contributing.md (#424)

**Commit:** a45bdc9  
**Author:** kwang-databricks  
**Date:** 2025-05-14

- Created preliminary Makefile with tox command integration
- Updated contributing documentation with new development processes
- Added support for Python 3.11 and removed Python 3.8
- Enhanced test and environment management commands
- Improved documentation for supported DBR versions
- Added dynamic virtualenv variable support