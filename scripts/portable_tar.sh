#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: portable_tar.sh [options] [--] PATH...

Create a cross-platform tar archive or tar stream without macOS-specific
metadata such as AppleDouble records or extended attributes/xattrs.
When supported by the local tar implementation, it also applies
--no-mac-metadata and --no-xattrs automatically.

Options:
  -C, --root DIR         Change to DIR before archiving paths.
  -f, --file PATH        Write archive to PATH instead of stdout.
      --stdout           Write archive to stdout (default).
  -z, --gzip             Gzip-compress the archive.
  -T, --files-from PATH  Read archive members from PATH (tar -T semantics).
      --exclude PATTERN  Exclude PATTERN from the archive. Repeatable.
  -h, --help             Show this help text.

Examples:
  portable_tar.sh -f repo.tar.gz -z -C /path/to/parent repo
  portable_tar.sh --gzip -C /path/to/parent repo | ssh host 'tar -xzf - -C /dest'
  portable_tar.sh -C /path/to/root -T filelist.txt --stdout | ssh host 'tar -xf - -C /dest'
EOF
}

supports_tar_flag() {
  case "${TAR_HELP_TEXT}" in
    *"$1"*) return 0 ;;
    *) return 1 ;;
  esac
}

mode="stdout"
output_path=""
root_dir=""
gzip_mode=0
tar_output="-"
tar_create_flag="-cf"
declare -a exclude_patterns=()
declare -a archive_paths=()
declare -a files_from_paths=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    -C|--root)
      root_dir="$2"
      shift 2
      ;;
    -f|--file)
      mode="file"
      output_path="$2"
      shift 2
      ;;
    --stdout)
      mode="stdout"
      shift
      ;;
    -z|--gzip)
      gzip_mode=1
      shift
      ;;
    -T|--files-from)
      files_from_paths+=("$2")
      shift 2
      ;;
    --exclude)
      exclude_patterns+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [ "$#" -gt 0 ]; do
        archive_paths+=("$1")
        shift
      done
      ;;
    -*)
      echo "ERROR: Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      archive_paths+=("$1")
      shift
      ;;
  esac
done

if [ "$mode" = "file" ] && [ -z "$output_path" ]; then
  echo "ERROR: --file requires a path." >&2
  exit 2
fi

if [ "$mode" = "file" ]; then
  tar_output="$output_path"
fi

if [ "$mode" = "file" ] && [ "$gzip_mode" -eq 0 ]; then
  case "$output_path" in
    *.tar.gz|*.tgz) gzip_mode=1 ;;
  esac
fi

if [ "${#archive_paths[@]}" -eq 0 ] && [ "${#files_from_paths[@]}" -eq 0 ]; then
  echo "ERROR: Provide at least one PATH or --files-from input." >&2
  usage >&2
  exit 2
fi

TAR_HELP_TEXT="$(tar --help 2>&1 || true)"
declare -a tar_args=()

if supports_tar_flag "--no-mac-metadata"; then
  tar_args+=(--no-mac-metadata)
fi

if supports_tar_flag "--no-xattrs"; then
  tar_args+=(--no-xattrs)
fi

if [ "${#exclude_patterns[@]}" -gt 0 ]; then
  for pattern in "${exclude_patterns[@]}"; do
    tar_args+=(--exclude "$pattern")
  done
fi

if [ -n "$root_dir" ]; then
  tar_args+=(-C "$root_dir")
fi

if [ "${#files_from_paths[@]}" -gt 0 ]; then
  for files_from_path in "${files_from_paths[@]}"; do
    tar_args+=(-T "$files_from_path")
  done
fi

if [ "$gzip_mode" -eq 1 ]; then
  tar_create_flag="-czf"
fi

tar_args+=("$tar_create_flag" "$tar_output")
if [ "${#archive_paths[@]}" -gt 0 ]; then
  tar_args+=("${archive_paths[@]}")
fi

exec env \
  COPYFILE_DISABLE=1 \
  COPY_EXTENDED_ATTRIBUTES_DISABLE=1 \
  tar "${tar_args[@]}"
