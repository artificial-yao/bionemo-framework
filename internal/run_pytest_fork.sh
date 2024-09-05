#!/bin/bash
test_tag="needs_fork"
test_dirs="tests/|examples/"  # (Py)Test directories separated by | for extended `grep`.
test_files=$(pytest --collect-only -q -m "${test_tag}" | grep -E "^(${test_dirs}).*.py" | sed 's/\.py.*$/.py/' | awk '{$1=$1;print}' | sort | uniq)
n_test_files=$(echo "$test_files" | wc -l)
echo "Forked PyTest Files: ${test_files}"
echo "Number of Forked PyTest Files: ${n_test_files}"
counter=1
# the overall test status collected from all pytest commands with test_tag
status=0

for testfile in $test_files; do
  rm -rf ./.pytest_cache/
  set -x
  if [[ $testfile != examples/* && $testfile != tests/* ]]; then
    testfile="tests/$testfile"
  fi
  echo "Running test ${counter} / ${n_test_files} : ${testfile}"

  pytest -m "${test_tag}" -vv --durations=0 --cov-append --cov=bionemo ${testfile}
  test_status=$?
  # Exit code 5 means no tests were collected: https://docs.pytest.org/en/stable/reference/exit-codes.html
  test_status=$(($test_status == 5 ? 0 : $test_status))
  # Updating overall status of tests
  status=$(($test_status > $status ? $test_status : $status))
  set +x
  ((counter++))
done

echo "Waiting for the tests to finish..."
wait

echo "Completed"

exit $status
