#!/usr/bin/env python
"""
Django's command-line utility for administrative tasks.
This script serves as the main entry point for running Django management commands.
"""
import os
import sys


def main():
    """
    Main function to run administrative tasks using Django's command-line utility.
    """

    # Set the default Django settings module to 'core.settings'
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
    try:
        # Try to import the execute_from_command_line function from Django's core management module
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        # Handle ImportError if Django is not installed or not available
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
        
    # Execute Django management commands from the command line arguments
    execute_from_command_line(sys.argv)

# Entry point: Run the main function if this script is executed directly
if __name__ == '__main__':
    main()
