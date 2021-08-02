
class PersistenceManager:
    def __init__(self, log_dir):
        self.log_dir = log_dir

        # make the directory if not there already.
        # make sure path stupidity is set up.
