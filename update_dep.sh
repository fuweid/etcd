#!/usr/bin/env bash

dump_module_deps() {
  local json_mod
  json_mod=$(go mod edit -json)

  local module
  if ! module=$(echo "${json_mod}" | jq -r .Module.Path); then
    return 255
  fi

  local require
  require=$(echo "${json_mod}" | jq -r '.Require')
  if [ "$require" == "null" ]; then
    return 0
  fi

  echo "$require" | jq -r '.[] | .Path+","+.Version+","+if .Indirect then " (indirect)" else "" end+",'"${module}"'"'
}

dump_module_deps
