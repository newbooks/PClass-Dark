#!/usr/bin/env python
"""Plot Swiss-Prot audits, excluding queries annotated with GO:0005515."""

from pclass_dark.swiss_audit_cli import main

if __name__ == "__main__":
    main()
