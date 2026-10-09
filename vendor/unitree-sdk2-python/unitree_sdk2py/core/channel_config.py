import os
import tempfile


# A shared /tmp/cdds.LOG fails as soon as another Linux user creates it first.
_CYCLONEDDS_LOG_FILE = os.path.join(
    tempfile.gettempdir(),
    f"cdds-{os.geteuid()}-{os.getpid()}.LOG",
)


ChannelConfigHasInterface = f'''<?xml version="1.0" encoding="UTF-8" ?>
    <CycloneDDS>
        <Domain Id="any">
            <General>
                <Interfaces>
                    <NetworkInterface name="$__IF_NAME__$" priority="default" multicast="default"/>
                </Interfaces>
            </General>
            <Tracing>
                <Verbosity>config</Verbosity>
            <OutputFile>{_CYCLONEDDS_LOG_FILE}</OutputFile>
        </Tracing>
        </Domain>
    </CycloneDDS>'''

ChannelConfigAutoDetermine = '''<?xml version="1.0" encoding="UTF-8" ?>
    <CycloneDDS>
        <Domain Id="any">
            <General>
                <Interfaces>
                    <NetworkInterface autodetermine=\"true\" priority=\"default\" multicast=\"default\" />
                </Interfaces>
            </General>
        </Domain>
    </CycloneDDS>'''
