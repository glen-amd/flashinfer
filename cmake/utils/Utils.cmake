macro(__flashinfer_option variable description value)
  if(NOT DEFINED ${variable})
    set(${variable}
        ${value}
        CACHE STRING ${description})
  endif()
endmacro()

macro(flashinfer_list_option variable description value)
  __flashinfer_option(${variable} "${description}" "${value}")
endmacro()

set(FLASHINFER_ALL_OPTIONS)

# ##############################################################################
# An option that the user can select. Can accept condition to control when
# option is available for user. Usage: tvm_option(<option_variable> "doc string"
# <initial value or boolean expression> [IF <condition>]) The macro snippet is
# copied from Apache TVM codebase.
macro(flashinfer_option variable description value)
  set(__value ${value})
  set(__condition "")
  set(__varname "__value")
  list(APPEND FLASHINFER_ALL_OPTIONS ${variable})
  foreach(arg ${ARGN})
    if(arg STREQUAL "IF" OR arg STREQUAL "if")
      set(__varname "__condition")
    else()
      list(APPEND ${__varname} ${arg})
    endif()
  endforeach()
  unset(__varname)
  if("${__condition}" STREQUAL "")
    set(__condition 2 GREATER 1)
  endif()

  if(${__condition})
    if("${__value}" MATCHES ";")
      # list values directly pass through
      __flashinfer_option(${variable} "${description}" "${__value}")
    elseif(DEFINED ${__value})
      if(${__value})
        __flashinfer_option(${variable} "${description}" ON)
      else()
        __flashinfer_option(${variable} "${description}" OFF)
      endif()
    else()
      __flashinfer_option(${variable} "${description}" "${__value}")
    endif()
  else()
    unset(${variable} CACHE)
  endif()
endmacro()
