# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import http.client
import os
import re
import urllib.parse
import warnings
from typing import Any

# Silence warning from ngcbase pkg_resources use
warnings.simplefilter("ignore", UserWarning)

# ruff: noqa: E402
import aiohttp
from fsspec.asyn import sync_wrapper
from fsspec.callbacks import DEFAULT_CALLBACK
from fsspec.implementations.http import HTTPFileSystem
from loguru import logger

try:
    from ngcbase.api import utils as rest_utils
    from ngcbase.api.configuration import Configuration
    from ngcsdk import Client
    from registry.api.models import ModelAPI
except ImportError:
    rest_utils = None
    Configuration = None
    Client = None
    ModelAPI = None

from earth2studio.utils.imports import check_extra_imports


async def get_client(**kwargs) -> aiohttp.ClientSession:  # type: ignore
    """Default aiohttp client"""
    return aiohttp.ClientSession(**kwargs)


@check_extra_imports("dlwp", ["ngcbase", "ngcsdk", "registry"])
class NGCModelFileSystem(HTTPFileSystem):
    """NGC model registry file system for pulling and opening files that are stored
    on both public and private model registries. Largely a wrapper ontop of fsspec
    HTTPFileSystem class. This works using a url with the following pattern:

    `ngc://models/<org_id/team_id/model_id>@<version>/<path/in/repo>`

    or if no team

    `ngc://models/<org_id/model_id>@<version>/<path/in/repo>`

    For example:
    `ngc://models/nvidia/modulus/sfno_73ch_small@0.1.0/sfno_73ch_small/config.json`

    Note
    ----
    For private registries use one of the standard authentication methods such as
    an environment variable (NGC_CLI_API_KEY) or ngc config file a NGC API key that has
    access to the respective model registry.

    - https://docs.ngc.nvidia.com/cli/script.html?highlight=ngc_cli_api
    - https://docs.ngc.nvidia.com/cli/cmd.html

    Parameters
    ----------
    block_size: int
        Blocks to read bytes; if 0, will default to raw requests file-like
        objects instead of HTTPFile instances
    simple_links: bool
        If True, will consider both HTML <a> tags and anything that looks
        like a URL; if False, will consider only the former.
    same_scheme: bool
        When doing ls/glob, if this is True, only consider paths that have
        http/https matching the input URLs.
    size_policy: this argument is deprecated
    client_kwargs: dict
        Passed to aiohttp.ClientSession, see
        https://docs.aiohttp.org/en/stable/client_reference.html
        For example, ``{'auth': aiohttp.BasicAuth('user', 'pass')}``
    get_client: Callable[..., aiohttp.ClientSession]
        A callable which takes keyword arguments and constructs
        an aiohttp.ClientSession. It's state will be managed by
        the HTTPFileSystem class.
    storage_options: key-value
        Any other parameters passed on to requests
    cache_type, cache_options: defaults used in open
    """

    def __init__(  # type: ignore
        self,
        block_size=None,
        simple_links=True,
        same_scheme=True,
        size_policy=None,
        cache_type="bytes",
        cache_options=None,
        asynchronous=False,
        loop=None,
        client_kwargs=None,
        get_client=get_client,
        encoded=True,
        **storage_options,
    ):
        self.client = Client()
        # If no org, then get the first org of this API key and use that
        if not Configuration(self.client).is_guest_mode:
            org_name = Configuration(self.client).org_name
            if org_name is None:
                org_names = Configuration(self.client).get_org_names()
                org_default = org_names.get(list(org_names.keys())[-1])
                org_name = org_default["org_name"]
                logger.warning(
                    f"API key found but no org found, using the org {org_name}"
                )

            self.client.configure(org_name=org_name)
        else:
            logger.warning(
                "Using NGC guest mode, which may fail due to unauthorized access. "
                + "Consider using a valid NGC API key and org"
            )
        self.model_api = ModelAPI(self.client)

        super().__init__(
            simple_links,
            block_size,
            same_scheme,
            size_policy,
            cache_type,
            cache_options,
            asynchronous,
            loop,
            client_kwargs,
            get_client,
            encoded,
            **storage_options,
        )
        # Bit of a hack to easily get a sync version of this function
        # https://github.com/fsspec/filesystem_spec/blob/master/fsspec/asyn.py#L275
        # open requires the creation of a HttpFile, easier to sync intercept this at the start
        mth = sync_wrapper(getattr(self, "_get_model_asset_url"), obj=self)
        setattr(self, "get_model_asset_url", mth)

    def _parse_ngc_uri(
        self, root: str
    ) -> tuple[str, str, str | None, str | None, str | None]:
        """Parse NGC url into components"""
        # ngc://models/<org_id/team_id/model_id>@<version>/<path/in/repo>`
        suffix = "ngc://models/"
        # The regex check
        pattern = re.compile(rf"{suffix}[\w-]+(/[\w-]+)?(/[\w-]+)?@[A-Za-z0-9.]+")
        if not pattern.match(root):
            raise ValueError(
                "Invalid URL, should be of form ngc://models/<org_id/team_id/model_id>@<version>\n"
                + f" got {root}"
            )
        root = root.replace(suffix, "")
        if len(root.split("@")[0].split("/")) == 3:
            (org, team, model_info) = root.split("/", 2)
            (name, model_info) = model_info.split("@", 1)
        elif len(root.split("@")[0].split("/")) == 2:
            (org, model_info) = root.split("/", 1)
            (name, model_info) = model_info.split("@", 1)
            team = None
        else:
            (name, model_info) = root.split("@", 1)
            org = None
            team = None

        # Extract filename / path if present
        if len(model_info.rstrip("/").split("/")) > 1:
            (version, filepath) = model_info.split("/", 1)
        else:
            version = model_info.rstrip("/")
            filepath = None

        return name, version, org, team, filepath

    def _get_ngc_model_url(
        self,
        name: str,
        version: str,
        org: str = None,
        team: str = None,
        filepath: str = None,
        authenticated_api: bool = True,
    ) -> str:
        # Authenticated API
        if authenticated_api:
            # APIs from registry/api/models.py but switched to use the async download function
            url = self.model_api.get_direct_download_URL(
                name, version, org=org, team=team
            )
            if filepath:
                url = f"{url}?path={filepath}"
        # Public API
        else:
            url = "https://api.ngc.nvidia.com/v2/models/"
            relative_urls = []
            if org:
                relative_urls += ["org", org]
            if team:
                relative_urls += ["team", team]
            relative_urls += [name, version, "files"]
            url = urllib.parse.urljoin(url, os.path.join(*relative_urls))
            if filepath:
                url = f"{url}?path={filepath}"
        return url

    async def _get_ngc(
        self, asset_url: str, headers: dict[str, str] = {}
    ) -> tuple[int, dict[str, Any] | None]:
        """Send get request to NGC and recieves asset reponse

        Parameters
        ----------
        url : str
            File/asset URL
        headers :  dict[str, str], optional
            Headers for AuthN GET request, by default {}

        Returns
        -------
        tuple[int, str]
            Status code and response JSON
        """
        # Based on: ngc/apps/ngc-cli/-/blob/main/ngcbpc/ngcbpc/transfer/async_download.py?ref_type=heads#L782
        # Get the direct download URL
        async with aiohttp.ClientSession() as session:
            # Use http filesystem encode URL
            dl_url_resp = await session.get(self.encode_url(asset_url), headers=headers)
            status = dl_url_resp.status
            if status == http.client.OK:
                direct_url_dict = await dl_url_resp.json()
                return status, direct_url_dict

            return status, None

    async def _get_model_asset_url(self, rpath: str) -> str:
        # Check if ngc url, if not just pass
        if not rpath.startswith("ngc://"):
            return rpath
        # Parse remote url
        name, version, org, team, filepath = self._parse_ngc_uri(rpath)
        # Create headers to determine if we have authn headers and point to private vs public APIs
        # Auth header API from ngcbase/transfer/async_download.py and ngcsdk/ngcsdk/__init__.py
        auth_header = self.model_api.client.authentication.auth_header(
            auth_org=self.client.config.org_name,
            auth_team=self.client.config.team_name,
        )
        headers = rest_utils.default_headers(auth_header)

        direct_url = None
        if "Authorization" in headers:
            api_type = "private"
            url = self._get_ngc_model_url(name, version, org, team, filepath, True)
            status, response = await self._get_ngc(url, headers)
            # Attempt authn with renew
            if status == http.client.UNAUTHORIZED:
                auth_header = self.model_api.client.authentication.auth_header(
                    auth_org=self.client.config.org_name,
                    auth_team=self.client.config.team_name,
                )
                headers = rest_utils.default_headers(auth_header)
                status, response = await self._get_ngc(url, headers)
        # No API headers created, so fall back to public access method
        else:
            api_type = "public"
            # Check to see if asset is there on NGC at all
            url = self._get_ngc_model_url(name, version, org, team, filepath, False)
            status, response = await self._get_ngc(url)

        # NGC will sent us a specific download URL for the file, extract this
        if status == http.client.OK and response is not None:
            if "urls" not in response or len(response["urls"]) == 0:
                raise http.client.HTTPException(
                    "NGC did not return a valid download URL in response"
                )
            direct_url = response["urls"][0]

        #  Do some graceful error catching down here...
        if status == http.client.UNAUTHORIZED:
            raise http.client.HTTPException(
                f"Unauthorized NGC API key for {api_type} model package"
            )
        if status == http.client.NOT_FOUND:
            raise http.client.HTTPException(
                f"Requested {api_type} model package {rpath} not found"
            )
        if not direct_url:
            raise http.client.HTTPException(
                f"Failed to get valid download URL for {api_type} model package"
            )

        return direct_url

    async def _expand_path(self, path, recursive=False, maxdepth=None):  # type: ignore
        # hhttps://github.com/fsspec/filesystem_spec/blob/master/fsspec/asyn.py#L864
        # TODO: Improve to allow download of multiple files
        # get -> expand_path -> ``glob`` or ``find``, which may in turn call ``ls``
        # To find possible files one needs to hit the /file/ root and pull out
        # possible URLS from the response
        # E.g. https://api.ngc.nvidia.com/v2/models/nvidia/modulus/sfno_73ch_small/versions/0.1.0/files/
        # could add pages via apps/ngc-cli/-/blob/main/ngcbpc/ngcbpc/transfer/async_download.py?ref_type=heads#L817
        # response['modelFiles'] lists possible files
        raise NotImplementedError("Glob / recursive patterns not supported yet")
        return await super()._expand_path(path, recursive, maxdepth)

    async def _glob(self, path, maxdepth=None, **kwargs):  # type: ignore
        raise NotImplementedError("Glob / recursive patterns not supported yet")
        return await super()._glob(path, maxdepth, **kwargs)

    async def _find(self, path, maxdepth=None, withdirs=False, **kwargs):  # type: ignore
        raise NotImplementedError("Glob / recursive patterns not supported yet")
        return await super()._find(path, maxdepth, withdirs, **kwargs)

    async def _ls_real(self, url, detail=True, **kwargs):  # type: ignore
        url = await self._get_model_asset_url(url)
        return await super()._ls_real(url, detail, **kwargs)

    async def _get_file(  # type: ignore
        self, rpath, lpath, chunk_size=5 * 2**20, callback=DEFAULT_CALLBACK, **kwargs
    ):
        # Calling .get will call get_file via the AbstractFileSystem
        # https://github.com/fsspec/filesystem_spec/blob/master/fsspec/spec.py#L916
        # Will get converted to sync function via Metaclass
        # https://github.com/fsspec/filesystem_spec/blob/master/fsspec/asyn.py#L931
        url = await self._get_model_asset_url(rpath)
        return await super()._get_file(
            url, lpath, chunk_size=chunk_size, callback=callback, **kwargs
        )

    def _open(  # type: ignore
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=None,  # XXX: This differs from the base class.
        cache_type=None,
        cache_options=None,
        size=None,
        **kwargs,
    ):
        path = self.get_model_asset_url(path)
        return super()._open(
            path,
            mode,
            block_size,
            autocommit,
            cache_type,
            cache_options,
            size,
            **kwargs,
        )
