% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/performance_measures.R
\name{Prec}
\alias{Prec}
\title{Precision or PPV}
\usage{
Prec(ct, multi.class = "macro")
}
\arguments{
\item{ct}{Confusion Matrix.}

\item{multi.class}{Should the results of each class be aggregated, and how?
Options: "none", "macro", "micro". (Defaults: "macro").}
}
\value{
PPV (a single value).
}
\description{
`Prec()` computes the Precision of PPV (Positive Predictive Value) between the output
of a classification model and the actual values of the target.
The precision of each class can be aggregated. Macro-precision is the average of the
precision of each classes. Micro-precision is the weighted average.
}
\examples{
y <- c(rep("a",3),rep("b",2))
y_pred <- c(rep("a",2),rep("b",3))
ct <- table(y,y_pred)
Prec(ct)
}
