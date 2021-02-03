import os
import random
import re
import sys
import math

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    #creating dictionary for output
    pdist = dict()
    
    #if the page is not empty
    if corpus[page]:
        #0.15 distributed equally over all pages in corpus
        for pge in corpus:
            pdist[pge] = (1-damping_factor)/len(corpus)
            #if pge is in another page, 0.85 distributed equally to all pges in the page
            if pge in corpus[page]:
                pdist[pge] += damping_factor/len(corpus[page])
    #if page is empt, 1 distributed equally over all pages
    else:
        for pge in corpus:
            pdist[pge] = 1/len(corpus)        
    
    return pdist
        


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    #Creating Output Dictionary with all pages in corpus as keys and
    #all values initially set to 0
    spageranks = dict.fromkeys(list(corpus.keys()), 0)
    
    #list for samples
    samples = []
    
    #1st sample picked at random
    samples.append(random.choice(list(corpus)))
    
    #adding 1/n to the value of the respective key or page every time it gets sampled
    spageranks[samples[0]] += 1/n 
    
    for i in range(1, n):
        #this line got a bit too long, it appends the ith sample, which is chosen from the transition model
        #taking into account the probablity of each page being selected
        samples.append((random.choices(list(transition_model(corpus, samples[i-1], damping_factor).keys()), weights = list(transition_model(corpus, samples[i-1], damping_factor).values()), k=1))[0])
        spageranks[samples[i]] += 1/n
    
    return spageranks
    


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    #creating 2 dictionaries one for comparision and output
    PR = dict.fromkeys(corpus, 1/len(corpus))
    PR_temp = dict.fromkeys(corpus, 1/len(corpus))
    
    #loop variable true when change in pagerank is greater than 0.001
    loop = True
    while loop:
      for p in PR:
        #summation part of formula
        summation = 0.0
        
        #1st term of the forrmula
        PR_temp[p] = (1-damping_factor)/len(corpus)
        
        #adding summation part for each incoming link to p
        for k,v in corpus.items():

          if p in v:
            summation += (PR[k]/len(v))
          if not v:
            summation += (PR[k]/len(corpus))

        PR_temp[p] += damping_factor*summation
        
      #condition for loop   
      for p in PR:
        if not math.isclose(PR[p],PR_temp[p], abs_tol = 0.001):
          loop = True
        else:
          loop = False
        
        PR[p] = PR_temp[p]


    return PR

if __name__ == "__main__":
    main()
