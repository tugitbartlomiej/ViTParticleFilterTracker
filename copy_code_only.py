#!/usr/bin/env python3
"""
Code-Only Copy Script
=====================

Selectively copies only code files from ViTParticleFilterTracker to 
VITParticleFilterTracker_OnlyCode while preserving directory structure.

Usage:
    python copy_code_only.py                    # Normal copy
    python copy_code_only.py --dry-run          # Preview without copying
    python copy_code_only.py --verbose          # Detailed logging
    python copy_code_only.py --dry-run --verbose # Preview with details

Author: Claude Code
Date: 2025-07-28
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import time

class CodeOnlyCopier:
    """
    Selectively copies only code-related files while preserving directory structure
    """
    
    def __init__(self, source_dir, dest_dir, dry_run=False, verbose=False):
        """
        Initialize the copier
        
        Args:
            source_dir: Source directory path
            dest_dir: Destination directory path  
            dry_run: If True, only preview without copying
            verbose: If True, show detailed logging
        """
        self.source_dir = Path(source_dir)
        self.dest_dir = Path(dest_dir)
        self.dry_run = dry_run
        self.verbose = verbose
        
        # Files to INCLUDE (code files)
        self.include_extensions = {
            # Python files
            '.py', '.pyx', '.pyi',
            # Scripts
            '.bat', '.cmd', '.sh', '.ps1',
            # SLURM scripts
            '.slurm',
            # Configuration files
            '.json', '.yaml', '.yml', '.toml', '.cfg', '.ini',
            # Documentation (selective)
            '.md'  # Will filter specific names
        }
        
        # Files to EXCLUDE (data/cache/models)
        self.exclude_extensions = {
            # Images
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp', '.ico',
            # Archives
            '.zip', '.tar', '.gz', '.rar', '.7z', '.bz2',
            # Models and data
            '.pt', '.pth', '.onnx', '.pkl', '.h5', '.safetensors',
            # Cache and temp
            '.pyc', '.pyo', '.log', '.out', '.err',
            # Text files (mostly annotations)
            '.txt',  # Will allow specific names like requirements.txt
            # Data files
            '.csv', '.xml', '.sqlite', '.db'
        }
        
        # Directories to EXCLUDE
        self.exclude_dirs = {
            '__pycache__', '.git', '.vscode', '.idea', 
            'node_modules', '.pytest_cache', '.mypy_cache',
            'logs', 'temp', 'tmp'
        }
        
        # Special files to INCLUDE (even if extension would exclude)
        self.include_filenames = {
            'requirements.txt', 'requirements-dev.txt', 'requirements-test.txt',
            'README.md', 'CLAUDE.md', 'LICENSE', 'Makefile',
            'Dockerfile', '.gitignore', '.gitattributes'
        }
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'files_copied': 0,
            'files_skipped': 0,
            'dirs_created': 0,
            'total_size_copied': 0,
            'total_size_skipped': 0,
            'errors': 0
        }
    
    def should_include_file(self, file_path):
        """
        Determine if a file should be included in the copy
        
        Args:
            file_path: Path object of the file
            
        Returns:
            bool: True if file should be included
        """
        filename = file_path.name.lower()
        extension = file_path.suffix.lower()
        
        # Check special include filenames first
        if filename in {name.lower() for name in self.include_filenames}:
            return True
        
        # Exclude by extension
        if extension in self.exclude_extensions:
            return False
        
        # Include by extension
        if extension in self.include_extensions:
            # Special handling for .md files (only specific ones)
            if extension == '.md':
                return filename in {'readme.md', 'claude.md', 'changelog.md', 
                                  'contributing.md', 'license.md'}
            return True
        
        # Default: exclude unknown extensions
        return False
    
    def should_include_dir(self, dir_path):
        """
        Determine if a directory should be processed
        
        Args:
            dir_path: Path object of the directory
            
        Returns:
            bool: True if directory should be processed
        """
        dir_name = dir_path.name.lower()
        return dir_name not in self.exclude_dirs
    
    def get_file_size(self, file_path):
        """Get file size safely"""
        try:
            return file_path.stat().st_size
        except (OSError, IOError):
            return 0
    
    def copy_file(self, src_file, dst_file):
        """
        Copy a single file with error handling
        
        Args:
            src_file: Source file path
            dst_file: Destination file path
            
        Returns:
            bool: True if successful
        """
        try:
            if not self.dry_run:
                # Create parent directory if it doesn't exist
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
            
            file_size = self.get_file_size(src_file)
            self.stats['total_size_copied'] += file_size
            
            if self.verbose:
                size_str = f"{file_size:,} bytes" if file_size > 0 else "unknown size"
                action = "WOULD COPY" if self.dry_run else "COPIED"
                print(f"  {action}: {src_file} -> {dst_file} ({size_str})")
            
            return True
            
        except Exception as e:
            print(f"ERROR copying {src_file}: {e}")
            self.stats['errors'] += 1
            return False
    
    def scan_files(self):
        """
        Scan source directory to count total files for progress bar
        
        Returns:
            int: Total number of files to process
        """
        total_files = 0
        for root, dirs, files in os.walk(self.source_dir):
            # Filter directories
            dirs[:] = [d for d in dirs if self.should_include_dir(Path(root) / d)]
            total_files += len(files)
        return total_files
    
    def copy_directory_structure(self):
        """
        Main method to copy directory structure with selective file copying
        """
        print(f"{'DRY RUN: ' if self.dry_run else ''}Copying code files from:")
        print(f"  Source: {self.source_dir}")
        print(f"  Destination: {self.dest_dir}")
        print()
        
        if not self.source_dir.exists():
            print(f"ERROR: Source directory does not exist: {self.source_dir}")
            return False
        
        # Create destination directory
        if not self.dry_run:
            self.dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Scan for progress bar
        if not self.verbose:
            print("Scanning files...")
            total_files = self.scan_files()
            pbar = tqdm(total=total_files, desc="Processing files", unit="files")
        
        # Walk through source directory
        for root, dirs, files in os.walk(self.source_dir):
            current_dir = Path(root)
            
            # Filter directories (modify dirs in-place to affect os.walk)
            dirs[:] = [d for d in dirs if self.should_include_dir(current_dir / d)]
            
            # Calculate relative path
            rel_path = current_dir.relative_to(self.source_dir)
            dest_dir = self.dest_dir / rel_path
            
            # Create directory structure
            if not self.dry_run and files:  # Only create dir if it will have files
                dest_dir.mkdir(parents=True, exist_ok=True)
                self.stats['dirs_created'] += 1
            
            # Process files in current directory
            for filename in files:
                src_file = current_dir / filename
                dst_file = dest_dir / filename
                
                self.stats['files_processed'] += 1
                
                if not self.verbose:
                    pbar.update(1)
                
                if self.should_include_file(src_file):
                    if self.copy_file(src_file, dst_file):
                        self.stats['files_copied'] += 1
                    else:
                        self.stats['files_skipped'] += 1
                else:
                    # File skipped due to filtering
                    file_size = self.get_file_size(src_file)
                    self.stats['total_size_skipped'] += file_size
                    self.stats['files_skipped'] += 1
                    
                    if self.verbose:
                        print(f"  SKIPPED: {src_file} ({src_file.suffix} not included)")
        
        if not self.verbose:
            pbar.close()
        
        return True
    
    def print_statistics(self):
        """Print final statistics"""
        print("\n" + "="*60)
        print("COPY STATISTICS")
        print("="*60)
        print(f"Files processed:    {self.stats['files_processed']:,}")
        print(f"Files copied:       {self.stats['files_copied']:,}")
        print(f"Files skipped:      {self.stats['files_skipped']:,}")
        print(f"Directories created: {self.stats['dirs_created']:,}")
        print(f"Errors:             {self.stats['errors']:,}")
        print()
        
        # Size statistics
        copied_mb = self.stats['total_size_copied'] / (1024 * 1024)
        skipped_mb = self.stats['total_size_skipped'] / (1024 * 1024)
        total_mb = copied_mb + skipped_mb
        
        print(f"Total size processed: {total_mb:.1f} MB")
        print(f"Size copied:         {copied_mb:.1f} MB ({copied_mb/total_mb*100:.1f}%)")
        print(f"Size skipped:        {skipped_mb:.1f} MB ({skipped_mb/total_mb*100:.1f}%)")
        print(f"Space saved:         {skipped_mb:.1f} MB")
        print()
        
        if self.dry_run:
            print("NOTE: This was a DRY RUN - no files were actually copied.")
        else:
            print(f"Code-only copy completed successfully!")
            print(f"Destination: {self.dest_dir}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Copy only code files while preserving directory structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python copy_code_only.py                    # Normal copy
    python copy_code_only.py --dry-run          # Preview without copying  
    python copy_code_only.py --verbose          # Detailed logging
    python copy_code_only.py --dry-run --verbose # Preview with details

Included file types:
    - Python files: .py, .pyx, .pyi
    - Scripts: .bat, .cmd, .sh, .ps1
    - Config: .json, .yaml, .yml, .toml, .cfg, .ini
    - Docs: README.md, CLAUDE.md, requirements*.txt

Excluded file types:
    - Images: .jpg, .png, .gif, etc.
    - Archives: .zip, .tar, .gz, etc.
    - Models: .pt, .pth, .onnx, .pkl, etc.
    - Cache: __pycache__, .pyc, logs, etc.
        """
    )
    
    parser.add_argument('--source', type=str, 
                       default='.',
                       help='Source directory (default: current directory)')
    parser.add_argument('--dest', type=str,
                       default='../ViTParticleFilterTracker_ForEden', 
                       help='Destination directory')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview what would be copied without actually copying')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed logging of each file operation')
    
    args = parser.parse_args()
    
    # Initialize copier
    copier = CodeOnlyCopier(
        source_dir=args.source,
        dest_dir=args.dest,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    # Perform copy
    start_time = time.time()
    success = copier.copy_directory_structure()
    end_time = time.time()
    
    # Print statistics
    copier.print_statistics()
    print(f"Processing time: {end_time - start_time:.1f} seconds")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())