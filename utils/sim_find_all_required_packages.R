generate_install_script <- function(path = ".", output_file = "install_packages.R") {
  # Find relevant files
  file_pattern <- "\\.R$|\\.Rmd$|\\.Rnw$|\\.qmd$"
  files <- list.files(path, pattern = file_pattern, recursive = TRUE, full.names = TRUE)

  # Extract packages from all files
  packages <- unique(unlist(lapply(files, function(file) {
    lines <- tryCatch(readLines(file, warn = FALSE), error = function(e) character(0))
    lib_calls <- stringr::str_match(lines, "library\\s*\\(\\s*['\"]?([[:alnum:]\\.]+)['\"]?\\s*\\)")[,2]
    req_calls <- stringr::str_match(lines, "require\\s*\\(\\s*['\"]?([[:alnum:]\\.]+)['\"]?\\s*\\)")[,2]
    pkg_colon <- unlist(lapply(stringr::str_match_all(lines, "([[:alnum:]\\.]+)::"), function(m) m[,2]), use.names = FALSE)
    pkg_triple_colon <- unlist(lapply(stringr::str_match_all(lines, "([[:alnum:]\\.]+):::"), function(m) m[,2]), use.names = FALSE)
    c(lib_calls, req_calls, pkg_colon, pkg_triple_colon)
  })))

  # Clean and deduplicate
  packages <- sort(unique(na.omit(packages)))

  # Get installed versions
  installed <- installed.packages()
  existing <- packages[packages %in% rownames(installed)]
  versions <- sapply(existing, function(pkg) as.character(packageVersion(pkg)))

  # Generate install_version_if_needed() calls
  install_lines <- paste0('install_version_if_needed("', existing, '", version = "', versions, '")')

  # Write to file
  writeLines(install_lines, output_file)
  message("Installation script written to: ", output_file)

  invisible(install_lines)
}

generate_install_script(path = ".")
