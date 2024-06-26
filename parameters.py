NEW_INSENT_TOKEN = "<t>"
ORD_A = 97


TEMPLATE_FEWNERD = {
    "None": "{ent} do/does not belong to any known entities.",
    "person-artist/author": "{ent} is an artist or author.",
    "person-actor": "{ent} is an actor.",
    "art-writtenart": "{ent} is a kind of writtenart.",
    "person-director": "{ent} is a director.",
    "person-other": "{ent} is a person, but not affiliated with following professions: actor, artist, athlete, author, director, politician, scholar, soldier.",
    "organization-other": "{ent} pertains to an organization that does not fall under the categories of company, educational institution, government, media, political party, religion, sports league, sports team, band or musical group.",
    "organization-company": "{ent} is a company",
    "organization-sportsteam": "{ent} is a sports team",
    "organization-sportsleague": "{ent} is a sports league",
    "product-car": "{ent} is a kind of car",
    "event-protest": "{ent} refers to a protest, uprising or revolution event",
    "organization-government/governmentagency": "{ent} refers to a government or governmental agency",
    "other-biologything": "{ent} is a special term about biology / life science.",
    "location-GPE": "{ent} is a kind of geopolitical entity",
    "location-other": "{ent} is a geographic locaton that does not fall under the categories of geopolitical entity, body of water, island, mountain, park, road, railway and transit.",
    "person-athlete": "{ent} is an athlete or coach.",
    "art-broadcastprogram": "{ent} is a broadcast program.",
    "product-other": "{ent} is a kind of product that does not fall under the categories of airplane, train, ship, car, weapon, food, electronic game and software.",
    "building-other": "{ent} is a kind of building that does not fall under the categories of airport, hospital, hotel, library, restaurant, sports facility and theater",
    "product-weapon": "{ent} is a kind of weapon.",
    "building-airport": "{ent} is an airport.",
    "building-sportsfacility": "{ent} is a sports facility building.",
    "person-scholar": "{ent} is a scholar.",
    "art-music": "{ent} is a music.",
    "event-other": "{ent} refers to some event except attack, election, natural disaster, protest, revolution and sports",
    "other-language": "{ent} is a kind of human language.",
    "other-chemicalthing": "{ent} is some special term about chemical science.",
    "art-film": "{ent} is a film.",
    "building-hospital": "{ent} is a hospital.",
    "other-law": "{ent} is a legal document, a term or a convention in legal sense.",
    "product-airplane": "{ent} is kind of airplane product.",
    "location-road/railway/highway/transit": "{ent} is a geographic position about roadways, railways, highways or public transit systems.",
    "person-soldier": "{ent} is a soldier",
    "location-mountain": "{ent} is geographic position about mountain.",
    "organization-education": "{ent} is an educational institute/organization.",
    "organization-media/newspaper": "{ent} is a media/newspaper organization.",
    "product-software": "{ent} is a software product.",
    "location-island": "{ent} is geographic position about island.",
    "location-bodiesofwater": "{ent} is geographic position situated near a body of water.",
    "building-library": "{ent} is a library.",
    "other-astronomything": "{ent} is a special term about astronomy.",
    "person-politician": "{ent} is a politician or lawyer or judge.",
    "building-hotel": "{ent} is a hotel building.",
    "product-game": "{ent} is a electronic game product.",
    "other-award": "{ent} is a kind of award.",
    "event-sportsevent": "{ent} refers to some event related to sports.",
    "organization-showorganization": "{ent} is a band or musical organization.",
    "other-educationaldegree": "{ent} is a kind of educational degree.",
    "building-theater": "{ent} is a theater.",
    "other-disease": "{ent} is a kind of disease.",
    "event-election": "{ent} is an event about election.",
    "organization-politicalparty": "{ent} is a political party/organization.",
    "other-currency": "{ent} is a kind of currency.",
    "event-attack/battle/war/militaryconflict": "{ent} is an event about attack, battle, war or military conflict.",
    "product-ship": "{ent} is a ship.",
    "building-restaurant": "{ent} is a restaurant.",
    "other-livingthing": "{ent} is a living animal/creature/organism.",
    "art-other": "{ent} is a work of art, but not belong to the categories of music, film, written art, broadcast or painting.",
    "event-disaster": "{ent} is a natural disaster event.",
    "organization-religion": "{ent} is a religious organization.",
    "other-medical": "{ent} refers to some kind of medicine.entity",
    "location-park": "{ent} is a park.",
    "other-god": "{ent} is a god in some legend/religious story.",
    "product-food": "{ent} is a kind of food.",
    "product-train": "{ent} is a kind of train(vehicle).",
    "art-painting": "{ent} is an art painting.",
}


TEMPLATE_ACE = {
    "None": "The word {evt} does not trigger any known event.",
    "Movement.Transport": "The word {evt} triggers a TRANSPORT event: an ARTIFACT (WEAPON or VEHICLE) or a PERSON is moved from one PLACE (GEOPOLITICAL ENTITY, FACILITY, LOCATION) to another.",
    "Personnel.Elect": "The word {evt} triggers an ELECT event which implies an election.",
    "Personnel.Start-Position": "The word {evt} triggers a START-POSITION event: a PERSON elected or appointed begins working for (or changes offices within) an ORGANIZATION or GOVERNMENT.",
    "Personnel.Nominate": "The word {evt} triggers a NOMINATE event: a PERSON is proposed for a position through official channels.",
    "Conflict.Attack": "The word {evt} triggers an ATTACK event: a violent physical act causing harm or damage.",
    "Personnel.End-Position": "The word {evt} triggers an END-POSITION event: a PERSON stops working for (or changes offices within) an ORGANIZATION or GOVERNMENT.",
    "Contact.Meet": "The word {evt} triggers a MEET event: two or more entities come together at a single location and interact with one another face-to-face.",
    "Life.Marry": "The word {evt} triggers a MARRY event: two people are married under the legal definition.",
    "Contact.Phone-Write": "The word {evt} triggers a PHONE-WRITE event: two or more people directly engage in discussion which does not take place 'face-to-face'.",
    "Transaction.Transfer-Money": "The word {evt} triggers a TRANSFER-MONEY event: giving, receiving, borrowing, or lending money when it is NOT in the context of purchasing something. ",
    "Justice.Sue": "The word {evt} triggers a SUE event: a court proceeding has been initiated for the purposes of determining the liability of a PERSON, ORGANIZATION or GEOPOLITICAL ENTITY accused of committing a crime or neglecting a commitment",
    "Conflict.Demonstrate": "The word {evt} triggers a DEMONSTRATE event: a large number of people come together in a public area to protest or demand some sort of official action. For eample: protests, sit-ins, strikes and riots.",
    "Business.End-Org": "The word {evt} triggers an END-ORG event: an ORGANIZATION ceases to exist (in other words, goes out of business).",
    "Life.Injure": "The word {evt} triggers an INJURE event: a PERSON gets/got injured whether it occurs accidentally, intentionally or even self-inflicted.",
    "Life.Die": "The word {evt} triggers a DIE event: a PERSON dies/died whether it occurs accidentally, intentionally or even self-inflicted.",
    "Justice.Arrest-Jail":  "The word {evt} triggers a DIE event: a PERSON is sent to prison.",
    "Transaction.Transfer-Ownership": "The word {evt} triggers a TRANSFER-OWNERSHIP event: The buying, selling, loaning, borrowing, giving, or receiving of artifacts or organizations by an individual or organization.",
    "Justice.Execute": "The word {evt} triggers an EXECUTE event: a PERSON is/was executed",
    "Justice.Trial-Hearing": "The word {evt} triggers a TRIAL-HEARING event: a court proceeding has been initiated for the purposes of determining the guilt or innocence of a PERSON, ORGANIZATION or GEOPOLITICAL ENTITY accused of committing a crime.",
    "Justice.Sentence": "The word {evt} triggers a SENTENCE event:  the punishment for the DEFENDANT is issued",
    "Life.Be-Born": "The word {evt} triggers a BE-BORN event: a PERSON is given birth to.",
    "Justice.Charge-Indict": "The word {evt} triggers a CHARGE-INDICT event: a PERSON, ORGANIZATION or GEOPOLITICAL ENTITY is accused of a crime",
    "Business.Start-Org": "The word {evt} triggers a START-ORG event: a new ORGANIZATION is created.",
    "Justice.Convict": "The word {evt} trigges a CONVICT event: a PERSON, ORGANIZATION or GEOPOLITICAL ENTITY is convicted whenever it has been found guilty of a CRIME.",
    "Business.Declare-Bankruptcy": "The word {evt} triggers a DECLARE-BANKRUPTCY event: an Entity officially requests legal protection from debt collection due to an extremely negative balance sheet.",
    "Justice.Release-Parole": "The word {evt} triggers a RELEASE-PAROLE event.",
    "Justice.Fine": "The word {evt} triggers a FINE event: a GEOPOLITICAL ENTITY, PERSON or ORGANIZATION get financial punishment typically as a result of court proceedings.",
    "Justice.Pardon": "The word {evt} triggers a PARDON event:  a head-of-state or their appointed representative lifts a sentence imposed by the judiciary.",
    "Justice.Appeal": "The word {evt} triggers a APPEAL event: the decision of a court is taken to a higher court for review",
    "Business.Merge-Org": "The word {evt} triggers a MERGE-ORG event: two or more ORGANIZATION Entities come together to form a new ORGANIZATION Entity. ",
    "Justice.Extradite": "The word {evt} triggers a EXTRADITE event.",
    "Life.Divorce": "The word {evt} triggers a DIVORCE event: two people are officially divorced under the legal definition of divorce.",
    "Justice.Acquit": "The word {evt} triggers a ACQUIT event: a trial ends but fails to produce a conviction.",   
}


TEMPLATE_TACREV = {
    "None": "{subj} has no known relations to {obj}",
    "per:stateorprovince_of_death": "{subj} died in the state or province {obj}",
    "per:title": "{subj} is a {obj}",
    "org:member_of": "{subj} is the member of {obj}",
    "per:other_family": "{subj} is the other family member of {obj}",
    "org:country_of_headquarters": "{subj} has a headquarter in the country {obj}",
    "org:parents": "{subj} has the parent company {obj}",
    "per:stateorprovince_of_birth": "{subj} was born in the state or province {obj}",
    "per:spouse": "{subj} is the spouse of {obj}",
    "per:origin": "{subj} has the nationality {obj}",
    "per:date_of_birth": "{subj} has birthday on {obj}",
    "per:schools_attended": "{subj} studied in {obj}",
    "org:members": "{subj} has the member {obj}",
    "org:founded": "{subj} was founded in {obj}",
    "per:stateorprovinces_of_residence": "{subj} lives in the state or province {obj}",
    "per:date_of_death": "{subj} died in the date {obj}",
    "org:shareholders": "{subj} has shares hold in {obj}",
    "org:website": "{subj} has the website {obj}",
    "org:subsidiaries": "{subj} owns {obj}",
    "per:charges": "{subj} is convicted of {obj}",
    "org:dissolved": "{subj} dissolved in {obj}",
    "org:stateorprovince_of_headquarters": "{subj} has a headquarter in the state or province {obj}",
    "per:country_of_birth": "{subj} was born in the country {obj}",
    "per:siblings": "{subj} is the siblings of {obj}",
    "org:top_members/employees": "{subj} has the high level member {obj}",
    "per:cause_of_death": "{subj} died because of {obj}",
    "per:alternate_names": "{subj} has the alternate name {obj}",
    "org:number_of_employees/members": "{subj} has the number of employees {obj}",
    "per:cities_of_residence": "{subj} lives in the city {obj}",
    "org:city_of_headquarters": "{subj} has a headquarter in the city {obj}",
    "per:children": "{subj} is the parent of {obj}",
    "per:employee_of": "{subj} is the employee of {obj}",
    "org:political/religious_affiliation": "{subj} has political affiliation with {obj}",
    "per:parents": "{subj} has the parent {obj}",
    "per:city_of_birth": "{subj} was born in the city {obj}",
    "per:age": "{subj} has the age {obj}",
    "per:countries_of_residence": "{subj} lives in the country {obj}",
    "org:alternate_names": "{subj} is also known as {obj}",
    "per:religion": "{subj} has the religion {obj}",
    "per:city_of_death": "{subj} died in the city {obj}",
    "per:country_of_death": "{subj} died in the country {obj}",
    "org:founded_by": "{subj} was founded by {obj}"
}