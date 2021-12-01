import sys
from pathlib import Path
import blobconverter
from difflib import get_close_matches
import depthai as dai


class BlobManager:
    """
    Manager class that handles MyriadX blobs.

    """

    _zooName = None
    _zooDir = None
    _blobPath = None
    _configPath = None
    _useZoo = False
    _useBlob = False
    _zooModels = []

    def __init__(
        self,
        blobPath: Path = None,
        configPath: Path = None,
        zooName: str = None,
        zooDir: Path = None,
    ):
        """
        Args:
            blobPath (pathlib.Path, Optional): Path to the compiled MyriadX blob file
            configPath (pathlib.Path, Optional): Path to model config file that is used to download the model
            zooName (str, Optional): Model name to be taken from model zoo
            zooDir (pathlib.Path, Optional): Path to model zoo directory
        """
        if blobPath is not None:
            self._blobPath = blobPath
            self._useBlob = True

        if zooDir is not None:
            self._zooDir = zooDir
            self._zooModels = [f.stem for f in zooDir.iterdir() if f.is_dir()]

        if zooName is not None:
            self._zooName = zooName
            if self._zooDir is not None:
                model_yml_path = self._zooDir / self._zooName / "model.yml"
                blob_paths = list((self._zooDir / self._zooName).glob("*.blob"))
                if model_yml_path.exists():
                    self._configPath = model_yml_path
                elif len(blob_paths) > 0:
                    self._blobPath = blob_paths[0]
                    self._useBlob = True
                else:
                    self._useZoo = True
            else:
                self._useZoo = True

        if configPath is not None:
            self._configPath = configPath

    def getBlob(self, shaves: int, openvinoVersion: dai.OpenVINO.Version):
        """
        This function is responsible for returning a ready to use MyriadX blob once requested.
        It will compile the model automatically using our online blobconverter tool. The compilation process will be
        ran only once, each subsequent call will return a path to previously compiled blob

        Args:
            shaves (int): Specify how many shaves the model will use. Range 1-16
            openvinoVersion (depthai.OpenVINO.Version): OpenVINO version which will be used to compile the MyriadX blob

        Returns:
            pathlib.Path: Path to compiled MyriadX blob

        Raises:
            SystemExit: If model name is not found in the zoo, this method will print all available onces and terminate
            RuntimeError: If conversion failed with unknown status
            Exception: If some unknown error will occur (reraise)
        """
        version = openvinoVersion.name.replace("VERSION_", "").replace("_", ".")
        if self._useBlob:
            return self._blobPath
        elif self._useZoo:
            try:
                self._blobPath = blobconverter.from_zoo(
                    name=self._zooName, shaves=shaves, version=version
                )
                self._useBlob = True
                return self._blobPath
            except Exception as e:
                if hasattr(e, "response") and hasattr(e.response, "status_code"):
                    if "not found in model zoo" in e.response.text:
                        allModels = set(self._zooModels + blobconverter.zoo_list())
                        suggested = get_close_matches(self._zooName, allModels)
                        if len(suggested) > 0:
                            print(
                                "Model {} not found in model zoo. Did you mean: {} ?".format(
                                    self._zooName, " / ".join(suggested)
                                ),
                                file=sys.stderr,
                            )
                        else:
                            print("Model {} not found in model zoo", file=sys.stderr)
                        raise SystemExit(1)
                    raise RuntimeError(
                        'Blob conversion failed with status {}! Error: "{}"'.format(
                            e.response.status_code, e.response.text
                        )
                    )
                else:
                    raise
        elif self._configPath is not None:
            name = self._configPath.parent.stem
            self._blobPath = blobconverter.from_config(
                name=name,
                path=self._configPath,
                version=version,
                data_type="FP16",
                shaves=shaves,
            )
            self._useBlob = True
            return self._blobPath
