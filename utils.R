#' Load Peaks from Data
#'
#' This function extracts peak information from a nested list and returns a structured data frame.
#'
#' @param data A nested list where the first level of names represent time windows, and the second level of names
#'        represent modes. Each mode contains a list with electrode, peak_time, and peak_amp.
#'
#' @return A data frame with columns: 'time_window', 'mode', 'electrode', 'peak_time', and 'peak_amp'.
#'         Each row represents a single peak measurement.
#'
#' @examples
#' data <- list('0-50ms' = list('positive' = list(electrode=1, peak_time=25, peak_amp=0.5),
#'                              'negative' = list(electrode=2, peak_time=30, peak_amp=0.6)))
#' load.peaks(data)
#'
#' @importFrom stats rbind
load.peaks <- function (data) {
  peaks <- data.frame()
  for (time in names(data)){
    for (mode in names(data[time][[1]])) {
      x <- unlist(data[time][[1]][mode], use.names = FALSE)
      x <- c(x, c(time, mode))

      peaks <- rbind(peaks, x)
      names(peaks) <- c('electrode', 'peak_time', 'peak_amp', 'time_window', 'mode')

    }
  }

  # rearange columns
  peaks <- peaks[, c("time_window", "mode", "electrode", "peak_time", "peak_amp")]

  return(peaks)
}

#' Load and install R packages.
#'
#' This function checks for the given packages in the current R installation.
#' If a package is not found, it installs it from the specified repository.
#' It then loads the packages into the current R session.
#'
#' @param package A character vector of package names.
#' @param repos The repository URL to install packages from.
#'        Default is the Goettingen (Germany) mirror.
#'
#' @return A logical vector indicating successful loading of packages.
load.package <- function(package, repos) {

  # list of packages missing
  missing <- package[!package %in% installed.packages()[, 'Package']]

  # check wich packages are not intalled and install them
  if (!is.null(missing)) {
    if (missing(repos)) {
      # use Goettingen (Germany) mirror as default
      repos <- 'https://ftp.gwdg.de/pub/misc/cran/'
    }
    install.packages(missing, dependencies = TRUE,
                     repos = repos)
  }

  # load all packages
  sapply(package, require, character.only = TRUE)
}

#' Create APA-styled HTML table using the gt package.
#'
#' This function uses the gt package to create an APA-styled table with the
#' specified appearance.
#'
#' @param x A data frame or table to be styled.
#' @param title A character string specifying the title of the table.
#'        Default is an empty space.
#'
#' @param stub A logical value to determine if row names should be used as stub.
#'        Default is TRUE.
#'
#' @return A gt object with the specified stylings.
apa <- function(x, title = " ", stub = T) {
  # get gt package for making html tables
  load.package('gt')

  gt(x, rownames_to_stub = stub) %>%
    tab_stubhead(label = "Predictor") %>%
    tab_options(
      table.border.top.color = "white",
      heading.title.font.size = px(16),
      column_labels.border.top.width = 3,
      column_labels.border.top.color = "black",
      column_labels.border.bottom.width = 3,
      column_labels.border.bottom.color = "black",
      stub.border.color = "white",
      table_body.border.bottom.color = "black",
      table.border.bottom.color = "white",
      table.width = pct(100),
      table.background.color = "white"
    ) %>%
    cols_align(align="center") %>%
    tab_style(
      style = list(
        cell_borders(
          sides = c("top", "bottom"),
          color = "white",
          weight = px(1)
        ),
        cell_text(
          align="center"
        ),
        cell_fill(color = "white", alpha = NULL)
      ),
      locations = cells_body(
        columns = everything(),
        rows = everything()
      )
    ) %>%
    #title setup
    tab_header(
      title = html("<i>", title, "</i>")
    ) %>%
    opt_align_table_header(align = "left")
}

#' Format values for presentation.
#'
#' This function formats the given value to be presented in reports or tables.
#' If the absolute value is less than 0.001, it returns '< 0.001'. Otherwise,
#' it rounds and formats the value according to the given parameters.
#'
#' @param value A numeric value to be formatted.
#' @param nsmall A non-negative integer giving the minimum number of digits to
#'        the right of the decimal point. Default is 3.
#'
#' @param simplify A logical value. If TRUE, removes the '<' and '= ' prefixes.
#'        Default is FALSE.
#'
#' @return A character string of the formatted value.
format.value <- function(value, nsmall = 3, simplify = TRUE) {

  if (abs(value) < 0.001) {
    print_value <- '< 0.001'
  } else {
    print_value <- paste0('= ' , format(round(value, digits = nsmall), nsmall = nsmall))
  }

  if (simplify) {
    print_value <- gsub('< |= ', '', print_value)
  }

  return(print_value)
}
