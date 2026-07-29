"""On-disk format versioning for molito HDF5 shards.

Every shard written by ``GraphBatch``, ``ProteinBatch`` or ``ComplexBatch`` carries two
attributes at the file root:

  * ``molito_format_version`` -- an integer describing the on-disk layout. This is the one
    with semantics: readers use it to decide whether they understand the file.
  * ``molito_version`` -- the package version that wrote the shard, informational only.

Why an integer rather than the package version: the layout changes far less often than the
package does, so bumping a separate counter keeps the compatibility check meaningful. Bump
``FORMAT_VERSION`` whenever a change would stop an older molito from reading a new file --
adding a required dataset, changing a dtype, or altering how offsets are computed. Adding an
*optional* dataset does not need a bump, since older readers use ``group.get(...)`` and treat
a missing dataset as absent (this is how ``chain_ids`` was added).

Shards written before versioning was introduced have no attributes at all. Those load as
``LEGACY_FORMAT_VERSION`` (0) and are still readable -- the layout did not change when the
stamp was added, so the stamp is the only difference.

A file whose version is *newer* than this package understands is refused outright, since the
alternative is reading nonsense from datasets that have moved.
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import h5py

# Bump when a layout change would stop older readers from working. See module docstring.
FORMAT_VERSION = 1

# Shards written before the stamp existed carry no attributes.
LEGACY_FORMAT_VERSION = 0

FORMAT_VERSION_ATTR = "molito_format_version"
PACKAGE_VERSION_ATTR = "molito_version"


def _package_version() -> str:
    try:
        return version("molito")
    except PackageNotFoundError:
        return "unknown"


def stamp_format(f: h5py.File) -> None:
    """Write the format and package version attributes onto a newly created shard."""

    f.attrs[FORMAT_VERSION_ATTR] = FORMAT_VERSION
    f.attrs[PACKAGE_VERSION_ATTR] = _package_version()


def check_format(f: h5py.File, save_file: str | Path | None = None) -> int:
    """Validate the format version of a shard being opened and return it.

    Args:
        f: Open HDF5 file to check.
        save_file: Path used in the error message, for context when a load fails.

    Returns:
        The shard's format version, or LEGACY_FORMAT_VERSION for pre-versioning shards.

    Raises:
        RuntimeError: If the shard was written by a newer molito than this one.
    """

    file_version = int(f.attrs.get(FORMAT_VERSION_ATTR, LEGACY_FORMAT_VERSION))

    if file_version > FORMAT_VERSION:
        written_by = f.attrs.get(PACKAGE_VERSION_ATTR, "unknown")
        location = f" ({save_file})" if save_file is not None else ""
        raise RuntimeError(
            f"Shard{location} uses molito format version {file_version}, but this molito "
            f"({_package_version()}) only understands up to {FORMAT_VERSION}. "
            f"The shard was written by molito {written_by} -- upgrade to read it."
        )

    return file_version
