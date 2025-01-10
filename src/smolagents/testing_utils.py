#!/usr/bin/env python
# coding=utf-8

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import os


def get_tests_dir(append_path=None):
    """
    Args:
        append_path: optional path to append to the tests dir path

    Return:
        The full path to the `tests` dir, so that the tests can be invoked from anywhere. Optionally `append_path` is
        joined after the `tests` dir the former is provided.

    """
    # this function caller's __file__
    caller__file__ = inspect.stack()[1][1]
    tests_dir = os.path.abspath(os.path.dirname(caller__file__))

    while not tests_dir.endswith("tests"):
        tests_dir = os.path.dirname(tests_dir)

    if append_path:
        return os.path.join(tests_dir, append_path)
    else:
        return tests_dir
