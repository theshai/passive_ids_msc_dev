import pandas as pd
import pytest

@pytest.fixture
def base_unsw():
    """
    Fixture to create a base UNSW dataset for testing.

    Returns:
        pd.DataFrame: A DataFrame representing the base UNSW dataset.
    """
    data = {
        'id': [1, 2, 3, 4, 5],
        'dur': [1.0, 2.0, 3.0, 4.0, 5.0],
        'proto': ["tcp", "udp", "icmp", "igmp", "gre"],
        'service': ["http", "ftp", "ssh", "dns", "smtp"],
        'state': ["FIN", "SYN", "RST", "ACK", "PSH"],
        'attck_cat': ["Normal", "DoS", "Probe", "R2L", "U2R"],
        'label': [0, 1, 0, 1, 0]
       
    }
    return pd.DataFrame(data)