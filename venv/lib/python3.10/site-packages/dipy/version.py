
"""
Module to expose more detailed version info for the installed `scipy`
"""
version = "1.10.0"
full_version = version
short_version = version.split('.dev')[0]
git_revision = "4eb161f32d7cf6100a656372ea8c740800ec2cc6"
release = 'dev' not in version and '+' not in version

if not release:
    version = full_version
