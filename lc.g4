grammar lc;

root: term EOF | macro_definition EOF;

term:
	OPEN term CLOSE					# parenthesis
	| term term						# application
	| LAMBDA parameters DOT term	# abstraction
	| LETTER						# letter
	| MACRO_NAME					# macro
	| term MACRO_SYMBOL term		# infixMacro;

macro_definition: (MACRO_SYMBOL | MACRO_NAME) MACRO_OPERATOR term;

parameters: (LETTER)+;

OPEN: '(';
CLOSE: ')';

DOT: '.';
LAMBDA: '\\' | 'λ';
LETTER: [a-z];

WS: [ \t\n\r]+ -> skip;

MACRO_OPERATOR: '≡' | '=';
MACRO_NAME: [A-Z][A-Z0-9]*;
MACRO_SYMBOL: ~[a-zA-Z];
