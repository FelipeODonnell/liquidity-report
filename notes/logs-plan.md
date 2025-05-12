# API Logging Implementation Plan

This document outlines a plan to implement API response logging for the CoinGlass API data collection tool. The goal is to track errors and issues with API responses without changing the existing data collection functionality.

## Objectives

1. Create a logging mechanism that records API response metadata, particularly errors
2. Store logs in a structured format for easy analysis
3. Ensure minimal impact on existing code
4. Keep logs separate from the actual collected data
5. Make logs easily accessible and readable

## Implementation Details

### 1. Create a Logging Module

Create a new `api_logger.py` module with the following functionality:

```python
import json
import logging
import os
from datetime import datetime

class APILogger:
    """
    Logger for API responses that focuses on errors and metadata.
    """
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.ensure_log_directory()
        self.setup_logger()
        
    def ensure_log_directory(self):
        """Create log directory if it doesn't exist."""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # Create date-based subdirectory
        self.current_date = datetime.now().strftime('%Y%m%d')
        self.date_dir = os.path.join(self.log_dir, self.current_date)
        if not os.path.exists(self.date_dir):
            os.makedirs(self.date_dir)
    
    def setup_logger(self):
        """Set up file and console logging."""
        # Create logger
        self.logger = logging.getLogger('api_logger')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        log_file = os.path.join(self.date_dir, 'api_responses.log')
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_response(self, api_file, url, response, exception=None):
        """
        Log API response metadata with a focus on errors.
        
        Args:
            api_file: The API file that made the request
            url: The URL that was requested
            response: The response object from requests library
            exception: Exception object if an error occurred
        """
        # Extract base filename
        base_file = os.path.basename(api_file)
        
        # Create log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "api_file": base_file,
            "url": url,
            "success": False if exception else True
        }
        
        # Add response info if available
        if response:
            log_entry["status_code"] = response.status_code
            log_entry["response_time"] = response.elapsed.total_seconds()
            
            # Try to parse response json
            try:
                response_data = response.json()
                if "code" in response_data:
                    log_entry["api_code"] = response_data.get("code")
                if "msg" in response_data:
                    log_entry["api_message"] = response_data.get("msg")
                
                # Check for error conditions in the response
                if response_data.get("code") != "0":  # Non-zero codes indicate API errors
                    self.logger.error(f"API Error in {base_file}: {response_data.get('msg', 'Unknown error')}")
                    log_entry["error"] = True
                    
                # Check if data exists and has content
                if "data" not in response_data or not response_data.get("data"):
                    self.logger.warning(f"No data returned in {base_file}")
                    log_entry["empty_data"] = True
            except Exception as e:
                log_entry["parse_error"] = str(e)
                self.logger.error(f"Failed to parse JSON response from {base_file}: {str(e)}")
        
        # Add exception details if an error occurred
        if exception:
            log_entry["exception"] = str(exception)
            log_entry["exception_type"] = type(exception).__name__
            self.logger.error(f"Exception in {base_file}: {str(exception)}")
        
        # Save detailed log entry to JSON file
        self.save_log_entry(log_entry, base_file)
        
        # Return success status
        return log_entry.get("success", False)
    
    def save_log_entry(self, log_entry, base_file):
        """Save log entry to a JSON file."""
        # Create a sanitized filename
        sanitized_name = base_file.replace('.py', '').replace('.', '_')
        log_file = os.path.join(self.date_dir, f"{sanitized_name}_log.json")
        
        # Append to existing log file or create new one
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r+') as f:
                    try:
                        logs = json.load(f)
                        if not isinstance(logs, list):
                            logs = [logs]
                    except json.JSONDecodeError:
                        logs = []
                    
                    logs.append(log_entry)
                    
                    # Move to beginning of file and write updated logs
                    f.seek(0)
                    json.dump(logs, f, indent=2)
                    f.truncate()
            else:
                with open(log_file, 'w') as f:
                    json.dump([log_entry], f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save log entry: {str(e)}")
```

### 2. Modify `report.py` to Use the Logger

Update the `run_api_files` function in `report.py` to log API responses:

```python
def run_api_files():
    """Run all non-commented API files with rate limiting."""
    if not API_FILES:
        print("No API files found. Run with --discover first to populate the list.")
        return
        
    current_date = create_date_folder()
    
    print(f"\nRunning API files for date: {current_date}\n")
    
    # Initialize API logger
    from api_logger import APILogger
    api_logger = APILogger()
    
    total_files = 0
    successful = 0
    
    # Initialize rate limiter (29 requests per minute)
    rate_limiter = RateLimiter(max_requests_per_minute=29)
    
    for api_file in API_FILES:
        # Skip commented files
        if api_file.startswith('#'):
            print(f"Skipping (commented out): {api_file}")
            continue
            
        total_files += 1
        
        # Apply rate limiting
        wait_time = rate_limiter.wait_if_needed()
        if wait_time > 0:
            print(f"Continuing after rate limit wait...")
        
        print(f"Running: {api_file}")
        try:
            # Create a special environment to capture API responses
            env = os.environ.copy()
            env['API_LOGGING_ENABLED'] = 'true'
            
            # Run the API file
            result = subprocess.run([sys.executable, api_file], check=True, 
                                   env=env, capture_output=True, text=True)
            
            # Log the API response
            api_logger.log_response(api_file, "Unknown URL", None)  # Basic logging
            
            print(f"✓ Completed: {api_file}")
            successful += 1
        except subprocess.CalledProcessError as e:
            print(f"✗ Error running: {api_file}")
            print(f"  Error details: {e}")
            
            # Log the error
            api_logger.log_response(api_file, "Unknown URL", None, exception=e)
    
    print(f"\nCompleted {successful} of {total_files} API requests.")
    print(f"Data saved to: data/{current_date}/")
    print(f"Logs saved to: logs/{current_date}/")
```

### 3. Create a Logger Class Wrapper

In a new module `log_wrapper.py`, create a wrapper that can be used to integrate logging into API files:

```python
import os
import json
import inspect
import functools
from datetime import datetime

class LogAPIResponse:
    """
    A context manager for logging API responses without modifying existing code.
    """
    def __init__(self, api_file=None, url=None):
        self.api_file = api_file or self._get_caller_filename()
        self.url = url
        self.response = None
        self.exception = None
        
    def _get_caller_filename(self):
        """Get the filename of the calling script."""
        frame = inspect.stack()[2]
        return frame.filename
        
    def __enter__(self):
        """Return self to capture the response."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log the response on exit."""
        if exc_type:
            self.exception = exc_val
            
        # Skip logging if not enabled
        if os.environ.get('API_LOGGING_ENABLED') != 'true':
            return False  # Don't suppress the exception
            
        # Import here to avoid circular imports
        from api_logger import APILogger
        logger = APILogger()
        logger.log_response(self.api_file, self.url, self.response, self.exception)
        
        return False  # Don't suppress the exception
        
    def capture(self, response):
        """Capture the response object."""
        self.response = response
        return response
```

### 4. Create a Log Analysis Script

Create a script to analyze and summarize API logs:

```python
#!/usr/bin/env python3
"""
Script to analyze API response logs.
"""

import os
import json
import argparse
from datetime import datetime
from collections import defaultdict

def analyze_logs(log_dir=None, date=None):
    """
    Analyze API logs for a specific date.
    
    Args:
        log_dir: The directory containing log files
        date: The date to analyze (format: YYYYMMDD)
    """
    # Use current date if none provided
    if not date:
        date = datetime.now().strftime('%Y%m%d')
        
    # Use default log directory if none provided
    if not log_dir:
        log_dir = "logs"
        
    # Create path to date directory
    date_dir = os.path.join(log_dir, date)
    if not os.path.exists(date_dir):
        print(f"No logs found for date: {date}")
        return
        
    # Find all log files
    log_files = [f for f in os.listdir(date_dir) if f.endswith('_log.json')]
    
    if not log_files:
        print(f"No log files found for date: {date}")
        return
        
    # Analyze logs
    stats = {
        "total_requests": 0,
        "successful": 0,
        "failed": 0,
        "errors": [],
        "empty_data": 0,
        "api_errors": 0,
        "slowest_requests": []
    }
    
    # Track API errors by file
    api_errors_by_file = defaultdict(int)
    empty_data_by_file = defaultdict(int)
    
    # Process each log file
    for log_file in log_files:
        file_path = os.path.join(date_dir, log_file)
        
        with open(file_path, 'r') as f:
            try:
                logs = json.load(f)
                if not isinstance(logs, list):
                    logs = [logs]
                    
                for log in logs:
                    stats["total_requests"] += 1
                    
                    if log.get("success", False):
                        stats["successful"] += 1
                    else:
                        stats["failed"] += 1
                        
                    if log.get("exception"):
                        stats["errors"].append({
                            "api_file": log.get("api_file"),
                            "error": log.get("exception")
                        })
                        
                    if log.get("empty_data"):
                        stats["empty_data"] += 1
                        empty_data_by_file[log.get("api_file")] += 1
                        
                    if log.get("api_code") != "0" and log.get("api_code") is not None:
                        stats["api_errors"] += 1
                        api_errors_by_file[log.get("api_file")] += 1
                        
                    if log.get("response_time"):
                        stats["slowest_requests"].append({
                            "api_file": log.get("api_file"),
                            "time": log.get("response_time"),
                            "url": log.get("url")
                        })
            except json.JSONDecodeError:
                print(f"Error parsing log file: {log_file}")
    
    # Sort slowest requests
    stats["slowest_requests"] = sorted(
        stats["slowest_requests"],
        key=lambda x: x.get("time", 0),
        reverse=True
    )[:5]  # Keep only top 5
    
    # Print summary
    print(f"\nAPI Response Log Summary for {date}")
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Successful: {stats['successful']} ({stats['successful']/stats['total_requests']*100:.1f}%)")
    print(f"Failed: {stats['failed']} ({stats['failed']/stats['total_requests']*100:.1f}%)")
    print(f"API Errors: {stats['api_errors']} ({stats['api_errors']/stats['total_requests']*100:.1f}%)")
    print(f"Empty Data: {stats['empty_data']} ({stats['empty_data']/stats['total_requests']*100:.1f}%)")
    
    if stats["errors"]:
        print("\nTop Errors:")
        for error in stats["errors"][:5]:  # Show top 5 errors
            print(f"  {error['api_file']}: {error['error']}")
            
    if api_errors_by_file:
        print("\nAPI Errors by File:")
        for api_file, count in sorted(api_errors_by_file.items(), key=lambda x: x[1], reverse=True):
            print(f"  {api_file}: {count}")
            
    if empty_data_by_file:
        print("\nEmpty Data by File:")
        for api_file, count in sorted(empty_data_by_file.items(), key=lambda x: x[1], reverse=True):
            print(f"  {api_file}: {count}")
            
    if stats["slowest_requests"]:
        print("\nSlowest Requests:")
        for req in stats["slowest_requests"]:
            print(f"  {req['api_file']}: {req['time']:.2f}s")
    
    # Save summary to file
    summary_file = os.path.join(date_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(stats, f, indent=2)
        
    print(f"\nDetailed summary saved to: {summary_file}")

def main():
    """Parse command line arguments and analyze logs."""
    parser = argparse.ArgumentParser(description="Analyze API response logs")
    parser.add_argument('--date', type=str, help='Date to analyze (format: YYYYMMDD)')
    parser.add_argument('--log-dir', type=str, help='Log directory')
    
    args = parser.parse_args()
    analyze_logs(args.log_dir, args.date)

if __name__ == "__main__":
    main()
```

## Integration Plan

1. **Stage 1: Basic Logging Functionality**
   - Create the `logs` directory
   - Implement the `api_logger.py` module
   - Update `report.py` to log basic API call information

2. **Stage 2: Enhanced Logging**
   - Implement the `log_wrapper.py` wrapper class
   - Add log analysis capabilities
   - Create script to review and summarize logs

3. **Stage 3: Integrate With Existing Codebase**
   - Implement a non-intrusive integration method using environment variables
   - Ensure all API calls are logged properly

## Usage

To enable logging, simply run `report.py` as usual:

```bash
python report.py
```

Logs will be stored in `logs/YYYYMMDD/` directory with the following files:
- `api_responses.log` - Main log file with all API responses
- `{api_file}_log.json` - JSON file with detailed information about each API call

To analyze logs for a specific date:

```bash
python analyze_logs.py --date 20250512
```

## Benefits

1. **Error Tracking**: Easily identify which API endpoints are experiencing issues
2. **Performance Monitoring**: Track response times to identify slow endpoints
3. **Data Quality**: Monitor for empty or invalid responses
4. **Troubleshooting**: Detailed logs for debugging API issues
5. **Non-intrusive**: Implementation doesn't require modifying existing API files

## Limitations and Future Improvements

1. **Limited Response Analysis**: The initial implementation only captures basic response metadata
2. **Manual Integration**: API files will need to use the wrapper for detailed logging
3. **Storage Requirements**: Logging all responses could require significant disk space over time

Future improvements could include:
- Automatic log rotation and cleanup
- Integration with monitoring tools
- Alert capabilities for persistent API errors
- Web interface for log review and analysis