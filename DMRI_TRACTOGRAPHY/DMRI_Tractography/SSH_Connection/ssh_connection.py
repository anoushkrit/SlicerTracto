import paramiko
import config

class SSHConnection:
    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SSHConnection, cls).__new__(cls)
            cls._instance.init()
        return cls._instance

    def init(self):
        """Establish the SSH connection."""
        self.hostname = config.hostname
        self.port = config.port
        self.username = config.username
        self.password = config.password
        self.private_key = config.private_key
        self.client = None
        self.ssh_status = False
        self.connect()

    def connect(self):
        """Establish the SSH connection."""
        try:
            # Create an SSH client
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Use password or private key authentication
            if self.password:
                self.client.connect(self.hostname, port=self.port, username=self.username, password=self.password)
            elif self.private_key:
                self.client.connect(self.hostname, port=self.port, username=self.username, key_filename=self.private_key)
            else:
                raise ValueError("Either password or private key must be provided")

            self.ssh_status = True
            logging.info(f"SSH connection established to {self.hostname}")
        except Exception as e:
            self.ssh_status = False
            logging.error(f"Failed to connect to {self.hostname}: {e}")

    def get_connection_instance(self):
        """Return the SSH client instance."""
        if self.client:
            return self.client
        else:
            logging.error("SSH connection is not established.")
            return None

    def get_connection_status(self):
        """Return the connection status."""
        return self.ssh_status

    def close(self):
        """Close the SSH connection."""
        if self.client:
            self.client.close()
            self.ssh_status = False
            logging.info(f"SSH connection to {self.hostname} closed.")
        else:
            logging.error("No active SSH connection to close.")
