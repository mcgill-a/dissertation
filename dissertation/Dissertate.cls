% -------------------------------------------------------------------
%  @LaTeX-class-file{
%     filename        = "Dissertate.cls",
%     version         = "2.0",
%     date            = "25 March 2014",
%     codetable       = "ISO/ASCII",
%     keywords        = "LaTeX, Dissertate",
%     supported       = "Send email to suchow@post.harvard.edu.",
%     docstring       = "Class for a dissertation."
% --------------------------------------------------------------------

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{Dissertate}[2014/03/25 v2.0 Dissertate Class]
\LoadClass[12pt, oneside, letterpaper]{book}

%
% Options
%
\RequirePackage{etoolbox}

% Line spacing: dsingle/ddouble
%   Whether to use single- or doublespacing.
\newtoggle{DissertateSingleSpace}
\togglefalse{DissertateSingleSpace}
\DeclareOption{dsingle}{
    \toggletrue{DissertateSingleSpace}
    \ClassWarning{Dissertate}{Single-spaced mode on.}
}
\DeclareOption{ddouble}{\togglefalse{DissertateSingleSpace}}

\ProcessOptions\relax

% Figure positioning
\usepackage{float}

% Line Spacing
%   Define two line spacings: one for the body, and one that is more compressed.
\iftoggle{DissertateSingleSpace}{
    \newcommand{\dnormalspacing}{1.2}
    \newcommand{\dcompressedspacing}{1.0}
}{
    \newcommand{\dnormalspacing}{1.8}
    \newcommand{\dcompressedspacing}{1.2}
}

% Block quote with compressed spacing
\let\oldquote\quote
\let\endoldquote\endquote
\renewenvironment{quote}
    {\begin{spacing}{\dcompressedspacing}\oldquote}
    {\endoldquote\end{spacing}}

% Itemize with compressed spacing
\let\olditemize\itemize
\let\endolditemize\enditemize
\renewenvironment{itemize}
    {\begin{spacing}{\dcompressedspacing}\olditemize}
    {\endolditemize\end{spacing}}

% Enumerate with compressed spacing
\let\oldenumerate\enumerate
\let\endoldenumerate\endenumerate
\renewenvironment{enumerate}
    {\begin{spacing}{\dcompressedspacing}\oldenumerate}
    {\endoldenumerate\end{spacing}}

% List of abbreviations
\usepackage{longtable}
\usepackage[acronym]{glossaries}

% Text layout.
\RequirePackage[width=5.75in, letterpaper]{geometry}
\usepackage{ragged2e}
\RaggedRight
\RequirePackage{graphicx}
% \usepackage{fixltx2e}
\parindent 12pt
\RequirePackage{lettrine}
\RequirePackage{setspace}
\RequirePackage{verbatim}

% Allow PDF's
\usepackage{pdfpages}

% Table graphics package
\usepackage{graphicx}
\usepackage{booktabs}

% Fonts.
\RequirePackage{color}
\RequirePackage{xcolor}
\usepackage{hyperref}
\RequirePackage{url}
\RequirePackage{amssymb}
\RequirePackage{mathspec}
\RequirePackage{upgreek} %for greek alphabet
% \setmathsfont(Digits,Latin,Greek)[Numbers={Proportional}]{EB Garamond}
% \setmathrm{EB Garamond}
\AtBeginEnvironment{tabular}{\addfontfeature{RawFeature=+tnum}}
\widowpenalty=300
\clubpenalty=300
\setromanfont[Ligatures={Common, TeX}, Scale=1.0]{EB Garamond}
\newfontfamily{\smallcaps}[RawFeature={+c2sc,+scmp}]{EB Garamond}
\setsansfont[Scale=MatchLowercase, BoldFont={Lato Bold}]{Lato Regular}
% \setmonofont[Scale=MatchLowercase]{Source Code Pro}
%Alex, change the "normalsize" below for both the label and the textfont. Options here: https://www.javatpoint.com/latex-fonts-size-and-styles
\RequirePackage[labelfont={bf,sf,large,singlespacing},
                textfont={sf,large,singlespacing},
                justification={justified,RaggedRight},
                singlelinecheck=false,
                margin=0pt,
                figurewithin=chapter,
                tablewithin=chapter]{caption}
\renewcommand{\thefootnote}{\fnsymbol{footnote}}
\RequirePackage{microtype}


% Headings and headers.
\RequirePackage{fancyhdr}
\RequirePackage[tiny, md, sc]{titlesec}
\setlength{\headheight}{15pt}
\pagestyle{plain}
\RequirePackage{titling}

% Front matter.
% \setcounter{tocdepth}{1}
\usepackage[titles]{tocloft}
\usepackage[titletoc]{appendix}

\renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}}
\renewcommand{\cftchapfont}{\normalsize \scshape}
\renewcommand\listfigurename{Listing of figures}
\renewcommand\listtablename{Listing of tables}

% Endmatter
\renewcommand{\setthesection}{\arabic{chapter}.A\arabic{section}}

% References.
\renewcommand\bibname{References}
\RequirePackage[comma, round, sort]{natbib}
\renewcommand{\bibnumfmt}[1]{[#1]}
\RequirePackage[palatino]{quotchap}
\renewcommand*{\chapterheadstartvskip}{\vspace*{-0.5\baselineskip}}
\renewcommand*{\chapterheadendvskip}{\vspace{1.3\baselineskip}}

% An environment for paragraph-style section.
\providecommand\newthought[1]{%
   \addvspace{1.0\baselineskip plus 0.5ex minus 0.2ex}%
   \noindent\textsc{#1}%
}

% Align reference numbers so that they do not cause an indent.
\newlength\mybibindent
\setlength\mybibindent{0pt}
\renewenvironment{thebibliography}[1]
    {\chapter*{\bibname}%
     \@mkboth{\MakeUppercase\bibname}{\MakeUppercase\bibname}%
     \list{\@biblabel{\@arabic\c@enumiv}}
          {\settowidth\labelwidth{\@biblabel{999}}
           \leftmargin\labelwidth
            \advance\leftmargin\dimexpr\labelsep+\mybibindent\relax\itemindent-\mybibindent
           \@openbib@code
           \usecounter{enumiv}
           \let\p@enumiv\@empty
           \renewcommand\theenumiv{\@arabic\c@enumiv}}
     \sloppy
     \clubpenalty4000
     \@clubpenalty \clubpenalty
     \widowpenalty4000%
     \sfcode`\.\@m}
    {\def\@noitemerr
      {\@latex@warning{Empty `thebibliography' environment}}
     \endlist}

% Some definitions.
\def\advisor#1{\gdef\@advisor{#1}}
\def\coadvisorOne#1{\gdef\@coadvisorOne{#1}}
\def\coadvisorTwo#1{\gdef\@coadvisorTwo{#1}}
\def\committeeInternal#1{\gdef\@committeeInternal{#1}}
\def\committeeInternalOne#1{\gdef\@committeeInternalOne{#1}}
\def\committeeInternalTwo#1{\gdef\@committeeInternalTwo{#1}}
\def\committeeExternal#1{\gdef\@committeeExternal{#1}}
\def\degreeyear#1{\gdef\@degreeyear{#1}}
\def\degreemonth#1{\gdef\@degreemonth{#1}}
\def\degreeterm#1{\gdef\@degreeterm{#1}}
\def\degree#1{\gdef\@degree{#1}}
\def\department#1{\gdef\@department{#1}}
\def\field#1{\gdef\@field{#1}}
\def\university#1{\gdef\@university{#1}}
\def\universitycity#1{\gdef\@universitycity{#1}}
\def\universitystate#1{\gdef\@universitystate{#1}}
\def\programname#1{\gdef\@programname{#1}}
\def\pdOneName#1{\gdef\@pdOneName{#1}}
\def\pdOneSchool#1{\gdef\@pdOneSchool{#1}}
\def\pdOneYear#1{\gdef\@pdOneYear{#1}}
\def\pdTwoName#1{\gdef\@pdTwoName{#1}}
\def\pdTwoSchool#1{\gdef\@pdTwoSchool{#1}}
\def\pdTwoYear#1{\gdef\@pdTwoYear{#1}}
% School name and location
\university{Edinburgh Napier University}
\universitycity{Edinburgh}
%\universitystate{New York}

% School colour found from university's graphic identity site:
% https://staff.napier.ac.uk/services/corporateaffairs/exchange/Documents/edinburgh-napier-brand-guidelines.pdf
\definecolor{SchoolColor}{rgb}{0.76, 0.08, 0.20} 

% napier red color for chapters
\definecolor{chaptergrey}{rgb}{0.76, 0.08, 0.20} % dialed back a little
\definecolor{midgrey}{rgb}{0.4, 0.4, 0.4}

\hypersetup{
    colorlinks,
    citecolor=black,
    filecolor=black,
    linkcolor=black,
    urlcolor=SchoolColor,
}

% Import acronyms
\input{frontmatter/abbreviations}
\makeglossaries

% Formatting guidelines found in:
% http://gsas.nyu.edu/docs/IO/4474/formattingguide.pdf

\renewcommand{\frontmatter}{
	\pagenumbering{roman}
	\input{frontmatter/personalize}
	\maketitle
%	\copyrightpage
	% \frontispiece logo page
%	\dedicationpage
	\autorship
	\gdpr
%	\acknowledgments
%	\preface
 	\abstractpage
	\tableofcontents

	% figure listing - required if you have any figures
	\listoffigures
	\phantomsection
	\addcontentsline{toc}{chapter}{List of figures}

	% table listing - required if you have any tables
	\listoftables
	\phantomsection
	\addcontentsline{toc}{chapter}{List of tables}
	\vspace*{\fill} \newpage
	
	% List of abbreviations
	\listofabbreviations
	

	\newpage
	\setcounter{page}{1}
	\pagenumbering{arabic}
}

\renewcommand{\maketitle}{
	\thispagestyle{empty}
	\vspace*{\fill}
	\vspace{100pt}
	\begin{center}
    {\Huge {\thetitle} \par}
	\vspace{25pt}
	by \\
	\vspace{15pt}
	\theauthor\\
	\vspace{135pt}
	\textsc{a dissertation submitted in partial fulfillment\\
	of the requirements for the degree of \\
	\vspace{8pt}
	\@degree \\
	\@department \\
	\@university \\
	\vspace{10pt}
	\@degreemonth, \@degreeyear}
	\end{center}
	\vspace*{\fill}
}

\newcommand{\copyrightpage}{
	\newpage
	\thispagestyle{empty}
	\vspace*{25pt}
	\begin{center}
	\scshape \noindent \small \copyright \  \small  \theauthor \\
	all rights reserved, \@degreeyear
	\end{center}
	\newpage
	\rm
}

\newcommand{\frontispiece}{
	\newpage
	\thispagestyle{empty}
	\vspace*{\fill}
	\begin{center}
    \includegraphics[width=45pt]{resources/NYUlogo.eps}
	\end{center}
	\vspace*{\fill}
}

\newcommand{\dedicationpage}{
	\phantomsection
	\addcontentsline{toc}{chapter}{Dedication}
	\newpage
	\setcounter{page}{4}
	\vspace*{\fill}
	\scshape \noindent \input{frontmatter/dedication}
	\vspace*{\fill}
	\newpage
	\rm
}

\newcommand{\autorship}{
	\phantomsection
	\addcontentsline{toc}{chapter}{Authorship Declaration}
	\chapter*{Authorship Declaration}
	\noindent
	\input{frontmatter/authorship}
	\vspace*{\fill} \newpage
}

\newcommand{\gdpr}{
	\phantomsection
	\addcontentsline{toc}{chapter}{General Data Protection Regulation Declaration}
	\chapter*{General Data Protection Regulation Declaration}
	\noindent
	\input{frontmatter/gdpr}
	\vspace*{\fill} \newpage
}

\newcommand{\acknowledgments}{
	\phantomsection
	\addcontentsline{toc}{chapter}{Acknowledgments}
	\chapter*{Acknowledgments}
	\noindent
	\input{frontmatter/acknowledgments}
	\vspace*{\fill} \newpage
}

\newcommand{\preface}{
	\phantomsection
	\addcontentsline{toc}{chapter}{Preface}
	\chapter*{Preface}
	\input{frontmatter/preface}
	\vspace*{\fill} \newpage
}

\newcommand{\abstractpage}{
	\phantomsection
	\addcontentsline{toc}{chapter}{Abstract}
	\chapter*{Abstract}
	\input{frontmatter/abstract}
	\vspace*{\fill} \newpage
}

\newcommand{\listofabbreviations}{
	\phantomsection
	\addcontentsline{toc}{chapter}{List of abbreviations}
% 	\chapter*{Listing of abbreviations}
	\printglossary[type=\acronymtype,title=Listing of abbreviations]
	\vspace*{\fill} \newpage
}


\renewcommand{\backmatter}{
    \bibliography{references}
    \addcontentsline{toc}{chapter}{References}
    \bibliographystyle{apalike2}
    \clearpage
    \begin{appendices}
        \include{appendices/appendixA}
        \include{appendices/appendixB}
        \include{appendices/appendixC}
        \include{appendices/appendixD}
    \end{appendices}
    \input{endmatter/personalize}
    \clearpage
    %\include{endmatter/colophon}
}


\usepackage{caption}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{hhline} 