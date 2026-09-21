"""Generation of data sets from the virtual plants: the truth side of the data path.

This package builds the plants from their full configuration, hidden physics included,
simulates them, validates the true trajectories, observes them, and hands
``Observations`` and known plant records to ``process_transfer.data``, which writes the
available branch without ever seeing the truth. It also keeps the private record of
each attempt and scans what was written for anything that should not be there.

Dependencies go one way: ``generation`` imports ``simulation``, ``measurement`` and
``data``; none of them imports ``generation``. Nothing here is imported from a script
under ``experiments/``, and no experiment is imported as a library.
"""
