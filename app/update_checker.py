import os
import platform
import logging
import hashlib
import ntpath
import posixpath
import re
import requests
import subprocess
import tempfile
from packaging import version
from PySide6.QtCore import QObject, Signal, QThread, QStandardPaths
from app.version import __version__

logger = logging.getLogger(__name__)
LAUNCHER_SOURCE_SUFFIX = "-windows-launcher-source.zip"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_RELEASE_DOWNLOAD_BYTES = 1_000_000_000


def select_release_asset_url(data: dict, system: str, latest_tag: str) -> str | None:
    assets = data.get("assets", []) or []
    if system == "Windows":
        expected_source_zip = (
            f"comic-translate-v{latest_tag}-windows-launcher-source.zip"
        ).lower()
        for asset in assets:
            if str(asset.get("name", "")).lower() == expected_source_zip:
                url = str(asset.get("browser_download_url", "") or "")
                if url:
                    return url
    elif system == "Darwin":
        for asset in assets:
            name = str(asset.get("name", "")).lower()
            if name.endswith(".dmg") or name.endswith(".pkg"):
                url = str(asset.get("browser_download_url", "") or "")
                if url:
                    return url
    return None


def parse_release_checksum(payload: str, filename: str) -> str:
    for raw_line in payload.splitlines():
        parts = raw_line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        digest, listed_name = parts
        listed_name = listed_name.lstrip("*").strip()
        if listed_name == filename and SHA256_RE.fullmatch(digest.lower()):
            return digest.lower()
    raise ValueError(f"SHA256SUMS.txt does not contain {filename}")


def checksum_url_for_release_asset(url: str) -> str:
    base, separator, _filename = url.rpartition("/")
    if not separator or not base:
        raise ValueError("Release asset URL has no parent path")
    return f"{base}/SHA256SUMS.txt"


def validate_download_filename(filename: str) -> str:
    if (
        not filename
        or filename != ntpath.basename(filename)
        or filename != posixpath.basename(filename)
        or filename in {".", ".."}
        or filename != filename.rstrip(". ")
        or any(ord(character) < 32 or character in '<>:"/\\|?*' for character in filename)
    ):
        raise ValueError("Unsafe release asset filename")
    return filename


class UpdateChecker(QObject):
    """
    Checks for product updates and handles downloaded release packages.
    """
    update_available = Signal(str, str, str)  # version, release_notes, download_url
    up_to_date = Signal()
    error_occurred = Signal(str)
    download_progress = Signal(int)
    download_finished = Signal(str) # file_path

    REPO_OWNER = "robinhood0107"
    REPO_NAME = "document-translate-myfix"

    def __init__(
        self,
        repo_owner: str | None = None,
        repo_name: str | None = None,
        *,
        allow_release_link_without_installer: bool = False,
    ):
        super().__init__()
        self._worker_thread = None
        self._worker = None
        self.repo_owner = repo_owner or self.REPO_OWNER
        self.repo_name = repo_name or self.REPO_NAME
        self.allow_release_link_without_installer = allow_release_link_without_installer

    def _safe_stop_thread(self):
        try:
            if self._worker_thread and self._worker_thread.isRunning():
                self._worker_thread.quit()
                self._worker_thread.wait()
        except RuntimeError:
            # The C++ object has been deleted
            pass
        except Exception as e:
            logger.error(f"Error stopping thread: {e}")
        self._worker_thread = None

    def check_for_updates(self):
        """Starts the check in a background thread."""
        self._safe_stop_thread()
            
        self._worker_thread = QThread()
        self._worker = UpdateWorker(
            self.repo_owner,
            self.repo_name,
            __version__,
            allow_release_link_without_installer=self.allow_release_link_without_installer,
        )
        self._worker.moveToThread(self._worker_thread)
        
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        
        self._worker.update_available.connect(self.update_available)
        self._worker.up_to_date.connect(self.up_to_date)
        self._worker.error.connect(self.error_occurred)
        
        self._worker_thread.started.connect(self._worker.run)
        self._worker_thread.start()

    def download_installer(self, url, filename):
        """Starts the download in a background thread."""
        self._safe_stop_thread()

        self._worker_thread = QThread()
        self._worker = DownloadWorker(url, filename)
        self._worker.moveToThread(self._worker_thread)
        
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        
        self._worker.progress.connect(self.download_progress)
        self._worker.finished_path.connect(self.download_finished)
        self._worker.error.connect(self.error_occurred)
        
        self._worker_thread.started.connect(self._worker.run)
        self._worker_thread.start()

    def run_installer(self, file_path):
        """Opens the downloaded installer or launcher-source package."""
        try:
            system = platform.system()
            if system == "Windows":
                # EXE/MSI packages launch normally. ZIP packages open in the
                # registered archive viewer so users can extract the source
                # bundle before running its first-run launcher.
                os.startfile(file_path)
            elif system == "Darwin": # macOS
                subprocess.Popen(["open", file_path])
        except Exception as e:
            self.error_occurred.emit(f"Failed to open release package: {e}")

    def shutdown(self):
        """Stops any active worker thread (best-effort)."""
        self._safe_stop_thread()
        self._worker_thread = None
        self._worker = None


class UpdateWorker(QObject):
    update_available = Signal(str, str, str)
    up_to_date = Signal()
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        owner,
        repo,
        current_version,
        *,
        allow_release_link_without_installer: bool = False,
    ):
        super().__init__()
        self.owner = owner
        self.repo = repo
        self.current_version = current_version
        self.allow_release_link_without_installer = allow_release_link_without_installer

    def _latest_release_or_tag(self) -> dict:
        release_url = f"https://api.github.com/repos/{self.owner}/{self.repo}/releases/latest"
        response = requests.get(release_url, timeout=10)
        if response.status_code != 404:
            response.raise_for_status()
            return response.json()

        tags_url = f"https://api.github.com/repos/{self.owner}/{self.repo}/tags?per_page=50"
        tags_response = requests.get(tags_url, timeout=10)
        tags_response.raise_for_status()
        candidates = []
        for tag in tags_response.json() or []:
            raw_name = str(tag.get("name", "") or "")
            normalized = raw_name.lstrip("v")
            try:
                parsed = version.parse(normalized)
            except Exception:
                continue
            if parsed.is_prerelease:
                continue
            candidates.append((parsed, raw_name))
        if not candidates:
            raise RuntimeError("Could not parse version from releases or tags.")
        _parsed, tag_name = max(candidates, key=lambda item: item[0])
        return {
            "tag_name": tag_name,
            "html_url": f"https://github.com/{self.owner}/{self.repo}/releases/tag/{tag_name}",
            "assets": [],
        }

    def run(self):
        try:
            data = self._latest_release_or_tag()
            
            latest_tag = data.get("tag_name", "").lstrip("v")
            if not latest_tag:
                 self.error.emit("Could not parse version from release.")
                 self.finished.emit()
                 return

            if version.parse(latest_tag) > version.parse(self.current_version):
                # Find appropriate asset
                system = platform.system()
                asset_url = select_release_asset_url(data, system, latest_tag)
                
                if asset_url or self.allow_release_link_without_installer:
                    self.update_available.emit(latest_tag, data.get("html_url", ""), asset_url)
                else:
                    self.error.emit(f"New version {latest_tag} available, but no installer found for your OS.")
            else:
                self.up_to_date.emit()

        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class DownloadWorker(QObject):
    progress = Signal(int)
    finished_path = Signal(str)
    error = Signal(str)
    finished = Signal()

    def __init__(self, url, filename):
        super().__init__()
        self.url = url
        self.filename = filename

    def run(self):
        partial_path = ""
        try:
            # Download to Downloads directory
            download_dir = QStandardPaths.writableLocation(QStandardPaths.DownloadLocation)
            if not download_dir:
                download_dir = os.path.join(os.path.expanduser("~"), "Downloads")
            
            # Fallback to temp if Downloads doesn't exist
            if not os.path.exists(download_dir):
                download_dir = tempfile.gettempdir()

            filename = validate_download_filename(self.filename)
            save_path = os.path.join(download_dir, filename)
            partial_path = f"{save_path}.partial"
            expected_sha256 = ""
            if filename.lower().endswith(LAUNCHER_SOURCE_SUFFIX):
                checksum_response = requests.get(
                    checksum_url_for_release_asset(self.url),
                    timeout=30,
                )
                checksum_response.raise_for_status()
                expected_sha256 = parse_release_checksum(
                    checksum_response.text,
                    filename,
                )

            response = requests.get(self.url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            if total_size < 0 or total_size > MAX_RELEASE_DOWNLOAD_BYTES:
                raise ValueError("Release package exceeds the download size limit")
            downloaded_size = 0
            digest = hashlib.sha256()

            with open(partial_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        digest.update(chunk)
                        downloaded_size += len(chunk)
                        if downloaded_size > MAX_RELEASE_DOWNLOAD_BYTES:
                            raise ValueError("Release package exceeds the download size limit")
                        if total_size > 0:
                            percent = int((downloaded_size / total_size) * 100)
                            self.progress.emit(percent)

            if expected_sha256 and digest.hexdigest().lower() != expected_sha256:
                raise ValueError("Downloaded release package SHA-256 does not match")
            os.replace(partial_path, save_path)
            partial_path = ""
            self.finished_path.emit(save_path)
            
        except Exception as e:
            self.error.emit(str(e))
        finally:
            if partial_path:
                try:
                    os.remove(partial_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.warning("Unable to remove partial update download: %s", partial_path)
            self.finished.emit()
