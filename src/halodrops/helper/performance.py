import logging

def check_launch_detect_in_afile(a_file:'str') -> bool:
    """Returns bool value of launch detect for a given A-file

    Given the path for an A-file, the function parses through the lines
    till it encounters the phrase 'Launch Obs Done?' and returns the
    boolean value for the 1 or 0 found after the '=' sign in the line with
    the aforementioned phrase.

    Parameters
    ----------
    a_file : str
        Path to A-file

    Returns
    -------
    bool
        True if launch is detected (1), else False (0)
    """
    
    with open(a_file, "r") as f:
        logging.info(f'Opened File: {a_file=}')
        lines = f.readlines()

        for i, line in enumerate(lines):
            if "Launch Obs Done?" in line:
                line_id = i
                logging.info(f'"Launch Obs Done?" found on line {line_id=}')
                break
        
        return bool(int(lines[line_id].split('=')[1]))
