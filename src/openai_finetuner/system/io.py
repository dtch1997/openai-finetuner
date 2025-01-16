import abc
from typing import Any

class IO(abc.ABC):
    """Abstract base class for IO operations."""
    
    @abc.abstractmethod
    def read(self, name: str) -> Any:
        """Read and return an object by name.
        
        Args:
            name: Name/identifier of the object to read
            
        Returns:
            The requested object
        """
        pass
        
    @abc.abstractmethod 
    def write(self, name: str, obj: Any) -> None:
        """Write an object with the given name.
        
        Args:
            name: Name/identifier to store the object under
            obj: The object to write
        """
        pass

class FileSystemIO(IO):
    """Handles reading and writing objects to the filesystem cache directory."""
    
    def __init__(self, subdir: str = ""):
        """Initialize with optional subdirectory under cache dir.
        
        Args:
            subdir: Optional subdirectory path under the cache directory
        """
        from ..constants import get_cache_dir
        self.base_dir = get_cache_dir()
        if subdir:
            self.base_dir = self.base_dir / subdir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def read(self, name: str) -> Any:
        """Read an object from a file in the cache directory.
        
        Args:
            name: Name of file to read from
            
        Returns:
            Contents of the file
        """
        file_path = self.base_dir / name
        if not file_path.exists():
            raise FileNotFoundError(f"No file found at {file_path}")
        with open(file_path, 'r') as f:
            return f.read()

    def write(self, name: str, obj: Any) -> None:
        """Write an object to a file in the cache directory.
        
        Args:
            name: Name of file to write to
            obj: Object to write to file
        """
        file_path = self.base_dir / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(str(obj))
