from __future__ import annotations
from dataclasses import dataclass
from antlr4 import *
from lcLexer import lcLexer
from lcParser import lcParser
from lcVisitor import lcVisitor

import logging
import random
import sys
import pydot
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
import uuid


@dataclass
class Abstraction:
    lamb: str
    param: Letter
    term: Term


@dataclass
class Application:
    leftTerm: Term
    rightTerm: Term


@dataclass
class Letter:
    val: str


Term = Abstraction | Application | Letter

macros_table = {}


# -------------BOT-------------#

TOKEN = '6240818386:AAHeRgLM2NLJx8CVQ4D2WlaNzLkgumWlDZg'
BOT_USERNAME = '@JuliaAChurchBot'

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="AChurchBot!\nBenvingut!"
    )


async def author_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="AChurchBot!\n@ Júlia Amenós Dien, 2023"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="""/start\n/author\n/help\n/macros\nExpressió λ-càlcul"""
    )


async def macros_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=botShowMacrosTable()
    )


async def lambdaExpression(update: Update, context: ContextTypes.DEFAULT_TYPE):
    expression: str = update.message.text
    input_stream = InputStream(expression)

    lexer = lcLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = lcParser(token_stream)
    tree = parser.root()

    if parser.getNumberOfSyntaxErrors() == 0:
        treeVisitor = TreeVisitor()
        t = treeVisitor.visitRoot(tree)
        if t:
            await update.message.reply_text(getParentizedTree(t))
            graph = pydot.Dot(graph_type='digraph')
            generateGraph(t, graph)
            image_data = graph.create_png()
            await update.message.reply_photo(image_data)
            await botEvaluate(t, update, context)


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')


# --------------VISITOR-------------#

class TreeVisitor(lcVisitor):

    def visitRoot(self, ctx):
        listC = list(ctx.getChildren())
        return self.visit(listC[0])

    def visitParenthesis(self, ctx):
        return self.visit(ctx.term())

    # Inicialment s'havia guardat la lambda escrita al input (\ o λ) per seguir el mateix estil que l'usuari al mostrar el resultat.
    # Donat que el motiu d'incloure \ era per facilitar l'escritura a l'usuari, s'ha acabat decidint mostrar sempre els resultats amb λ.
    def visitAbstraction(self, ctx):
        # lamb = ctx.LAMBDA().getText()
        lamb = 'λ'
        param = self.visit(ctx.parameters())
        term = self.visit(ctx.term())
        # Evitar propagar errors si s'utilitza una macro que no està previament definida.
        if param is None or term is None:
            return None
        # Currification
        for p in param[::-1]:
            term = Abstraction(lamb=lamb, param=Letter(p), term=term)
        return term

    def visitApplication(self, ctx):
        [t1, t2] = list(ctx.getChildren())
        leftTerm = self.visit(t1)
        rightTerm = self.visit(t2)
        if leftTerm is None or rightTerm is None:
            return None
        return Application(leftTerm=leftTerm, rightTerm=rightTerm)

    def visitLetter(self, ctx):
        return Letter(ctx.getText())

    def visitParameters(self, ctx):
        return ctx.getText()

    def visitMacro(self, ctx):
        name = ctx.MACRO_NAME().getText()
        if name in macros_table:
            return macros_table[name]
        print(f"Error: no existing macro: {name}")
        return None

    def visitInfixMacro(self, ctx):
        [t1, _, t2] = list(ctx.getChildren())
        term1 = self.visit(t1)
        term2 = self.visit(t2)
        name = ctx.MACRO_SYMBOL().getText()
        if name in macros_table:
            macro = macros_table[name]
            return Application(Application(macro, term1), term2)
        print(f"Error: no existing macro: {name}")
        return None

    def visitMacro_definition(self, ctx):
        name = ctx.MACRO_NAME().getText() if ctx.MACRO_NAME() else ctx.MACRO_SYMBOL().getText()
        term = self.visit(ctx.term())
        macros_table[name] = term
        return


# ---------------INTERPRETER---------------#

def showParentizedTree(a: Term):
    print("Arbre:")
    print(getParentizedTree(a))


# Returns a string representation of the given term with parentheses.
def getParentizedTree(a: Term):
    match a:
        case Abstraction(l, p, t):
            return (f"({l}{getParentizedTree(p)}.{getParentizedTree(t)})")
        case Application(t1, t2):
            str1 = getParentizedTree(t1)
            str2 = getParentizedTree(t2)
            return (f"({str1+str2})")
        case Letter(val):
            return val


# Performs the reduction replacement for the given term
# val (parameter to change), newVal (replacement)
def reduce(a: Term, val: Letter, newVal: Term):
    match a:
        case Abstraction(l, p, t):
            return Abstraction(l, p, reduce(t, val, newVal))
        case Application(l, r):
            newLeft = reduce(l, val, newVal)
            newRight = reduce(r, val, newVal)
            return Application(newLeft, newRight)
        case Letter(letter):
            if letter == val.val:
                return newVal
            return a


# Returns a beta-reduction of the given term
def beta_Reduction(a: Term):
    match a:
        case Application(l, r):
            if isinstance(l, Abstraction):
                convertedTerm = alpha_Conversion(l, r)
                param = convertedTerm.param
                term = convertedTerm.term
                currentTerm = reduce(term, param, r)
                print("β-reducció:")
                print(
                    f"{getParentizedTree(Application(convertedTerm, r))} → {getParentizedTree(currentTerm)}")
                return currentTerm

            else:
                reduced_left = beta_Reduction(l)
                # Permet que sempre es faci primer la beta-reducció més externa.
                if reduced_left == l:
                    reduced_right = beta_Reduction(r)
                    return Application(reduced_left, reduced_right)
                return Application(reduced_left, r)

        case Abstraction(lmb, p, t):
            return Abstraction(lmb, p, beta_Reduction(t))
    return a


async def botBetaReduction(a: Term, update: Update, context: ContextTypes.DEFAULT_TYPE):
    match a:
        case Application(l, r):
            if isinstance(l, Abstraction):
                convertedTerm = await botAlphaConversion(l, r, update, context)
                param = convertedTerm.param
                term = convertedTerm.term
                currentTerm = reduce(term, param, r)
                await update.message.reply_text(f"β-reducció: \n{getParentizedTree(Application(convertedTerm, r))} → {getParentizedTree(currentTerm)}")
                return currentTerm

            else:
                reduced_left = await botBetaReduction(l, update, context)
                if reduced_left == l:
                    reduced_right = await botBetaReduction(r, update, context)
                    return Application(reduced_left, reduced_right)
                return Application(reduced_left, r)

        case Abstraction(lmb, p, t):
            return Abstraction(lmb, p, await botBetaReduction(t, update, context))
    return a


# Returns the set of free variables in the given term
def checkFreeVariables(a: Term):
    match a:
        case Letter(letter):
            return {letter}
        case Application(l, r):
            return checkFreeVariables(l) | checkFreeVariables(r)
        case Abstraction(_, p, t):
            return checkFreeVariables(t) - checkFreeVariables(p)


# Returns the set of bound variables in the given term
def checkBoundVariables(a: Term):
    match a:
        case Letter(_):
            return set()
        case Application(l, r):
            return checkBoundVariables(l) | checkBoundVariables(r)
        case Abstraction(_, p, t):
            return {p.val} | checkBoundVariables(t)


# Generator of random variable names that are not used in a specific lambda expression.
# Unavailable is the set of variable names already in the expression..
def getRandomVariable(unavailable):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    available = set(alphabet) - set(unavailable)
    while True:
        yield random.choice(list(available))


# Substitutes occurrences of a variable in the term with a new variable.
# var (string to be replaced), newVar (string to substitute).
def convert(term: Term, var: str, newVar: str):
    match term:
        case Abstraction(l, p, t):
            if p.val == var:
                return Abstraction(l, Letter(newVar), convert(t, var, newVar))
            return Abstraction(l, p, convert(t, var, newVar))
        case Application(l, r):
            newLeft = convert(l, var, newVar)
            newRight = convert(r, var, newVar)
            return Application(newLeft, newRight)
        case Letter(letter):
            if letter == var:
                return Letter(newVar)
            return term


# Returns the given term with all the alpha_conversions that were needed.
def alpha_Conversion(l: Term, r: Term):
    boundVars = checkBoundVariables(l)
    freeVars = checkFreeVariables(r)

    # Set de variables que en les que es produeix conflicte.
    varsToConvert = boundVars & freeVars
    unusedVars = getRandomVariable(boundVars | freeVars)

    term = l
    for var in varsToConvert:
        newVar = next(unusedVars)
        convertedTerm = convert(term, var, newVar)
        if convertedTerm != term:
            print(f"α-conversió: {var} → {newVar}")
            print(f"{getParentizedTree(term)} → {getParentizedTree(convertedTerm)}")
            term = convertedTerm
    return term


async def botAlphaConversion(l: Term, r: Term, update: Update, context: ContextTypes.DEFAULT_TYPE):
    boundVars = checkBoundVariables(l)
    freeVars = checkFreeVariables(r)

    varsToConvert = boundVars & freeVars
    unusedVars = getRandomVariable(boundVars | freeVars)

    term = l
    for var in varsToConvert:
        newVar = next(unusedVars)
        convertedTerm = convert(term, var, newVar)
        if convertedTerm != term:
            await update.message.reply_text(f"α-conversió: {var} → {newVar}\n{getParentizedTree(term)} → {getParentizedTree(convertedTerm)}")
            term = convertedTerm
    return term


# Shows the table with all the current macros.
def showMacrosTable():
    for name, term in macros_table.items():
        print(f"{name} ≡ {getParentizedTree(term)}")


def botShowMacrosTable():
    message = ""
    for name, term in macros_table.items():
        message += f"{name} ≡ {getParentizedTree(term)}\n"
    print(message)
    return message


# Evaluates a lambda expression, applying alpha-conversions and beta-reductions.
# If more than 'limit' operations are needed to achieve the normal form, shows an error.
def evaluate(a: Term, limit=10):
    iter = 0
    previousTerm = a
    converted = False

    while iter < limit:
        result = beta_Reduction(previousTerm)
        if result != previousTerm:
            converted = True
        if converted and result == previousTerm:
            break
        previousTerm = result
        iter += 1

    print("Resultat:")
    print(getParentizedTree(result) if converted else "Nothing")
    print()


async def botEvaluate(a: Term, update: Update, context: ContextTypes.DEFAULT_TYPE, limit=10):
    iter = 0
    previousTerm = a
    converted = False

    while iter < limit:
        result = await botBetaReduction(previousTerm, update, context)
        if result != previousTerm:
            converted = True
        if converted and result == previousTerm:
            break
        previousTerm = result
        iter += 1

    await update.message.reply_text(f"Result: {getParentizedTree(result)}" if converted else "Result: Nothing, it can not be resolved.")
    resultGraph = pydot.Dot(graph_type='digraph')
    generateGraph(result, resultGraph)
    image_data = resultGraph.create_png()
    await update.message.reply_photo(image_data)


# ------ GRAPH GENERATION ------#

# Creates a random and unique identificator for a node.
def generate_unique_node():
    return str(uuid.uuid1())


# Generates the graph of a given term.
def generateGraph(a: Term, graph, shape='plaintext', boundVariables={}):
    match a:
        case Abstraction(lmb, p, t):
            symbol_node = "λ" + p.val
            node = pydot.Node(name=generate_unique_node(),
                              label=symbol_node, shape=shape)
            graph.add_node(node)

            boundVariables[p.val] = node

            body_node = generateGraph(t, graph, boundVariables=boundVariables)

            graph.add_edge(pydot.Edge(node, body_node))

        case Application(l, r):
            node = pydot.Node(name=generate_unique_node(),
                              label="@", shape=shape)
            graph.add_node(node)

            left_node = generateGraph(l, graph, boundVariables=boundVariables)
            right_node = generateGraph(r, graph, boundVariables=boundVariables)

            graph.add_edge(pydot.Edge(node, left_node))
            graph.add_edge(pydot.Edge(node, right_node))

        case Letter(letter):
            node = pydot.Node(name=generate_unique_node(),
                              label=letter, shape=shape)
            graph.add_node(node)

            for boundVar, boundNode in boundVariables.items():
                if boundVar == letter:
                    graph.add_edge(pydot.Edge(node, boundNode, style='dotted'))
                    break

    return node


# ------ MAIN ------#

if __name__ == '__main__':
    args = sys.argv

    # COMMAND LINE INTERPRETER
    if len(args) == 1 or len(args) == 2 and args[1] == '1':
        while True:
            input_stream = InputStream(input('? '))

            lexer = lcLexer(input_stream)
            token_stream = CommonTokenStream(lexer)
            parser = lcParser(token_stream)
            tree = parser.root()

            if parser.getNumberOfSyntaxErrors() == 0:
                treeVisitor = TreeVisitor()
                t = treeVisitor.visitRoot(tree)
                if not t:
                    showMacrosTable()
                else:
                    showParentizedTree(t)
                    evaluate(t)

    # BOT INTERPRETER
    elif len(args) == 2 and args[-1] == '2':
        app = ApplicationBuilder().token(TOKEN).build()

        # Commands
        app.add_handler(CommandHandler('start', start_command))
        app.add_handler(CommandHandler('author', author_command))
        app.add_handler(CommandHandler('help', help_command))
        app.add_handler(CommandHandler('macros', macros_command))

        # Messages
        app.add_handler(MessageHandler(filters.TEXT, lambdaExpression))

        # Errors
        app.add_error_handler(error)

        app.run_polling(poll_interval=2)

    else:
        print(
            "Usage: python3 achurch.py [num]\n   num: 1 (Command line interpreter) / 2 (telegram bot interpreter)")
