"""Shared code of the dating system, used by the scripts, the tests and the MCP server.

    statements  the parser (step 1), the rules that give each verdict (step 2)
                and the scoring of the candidate years (step 3)
    clauses     the reviewed clauses, read from the committed clause tables
    cal         the Babylonian calendar of Parker and Dubberstein
    years       the 701 candidate years and the conventions for year numbers
    sky         positions from JPL DE441, and the Babylonian zodiac
    eclipses    the eclipse catalogs and the visibility of each eclipse at Babylon
    eras        year statements inside a tablet, which are read only after dating
"""
