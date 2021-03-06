DROP TABLE "STAPUSER"."STAP_NPS_COMMENTS_ALL";
CREATE TABLE "STAPUSER"."STAP_NPS_COMMENTS_ALL" 
   (	"COMM_ID" raw(16) default sys_guid() primary key,
	"COMMENTS_DATE" VARCHAR2(200),
	"SITEID" VARCHAR2(100), 
	"SITEURL" VARCHAR2(100), 
	"SITETYPE" VARCHAR2(100), 
	"RELVERSION" VARCHAR2(100), 
	"CLIVERSION" VARCHAR2(100), 
	"DOMAINNAME" VARCHAR2(100), 
	"NPS_SCORE" VARCHAR2(100), 
	"COMMENTS" VARCHAR2(4000), 
	"JOIN_ISSUE" VARCHAR2(3000), 
	"AUDIO_ISSUE" VARCHAR2(3000), 
	"VIDEO_ISSUE" VARCHAR2(3000),
	"SHARING_ISSUE" VARCHAR2(3000),
	"OTHER_ISSUE" VARCHAR2(3000),
	"VC_CATEGORY" VARCHAR2(100),
	"COMM_WL" VARCHAR2(4000)
   );


DROP TABLE "STAPUSER"."STAP_NPS_TOPICS_ALL";
CREATE TABLE "STAPUSER"."STAP_NPS_TOPICS_ALL"
   (	"NPS_TOPIC_ID" VARCHAR2(100) NOT NULL ENABLE,
	"NPS_VERSION" VARCHAR2(100),
	"TOPIC_SENTIMENT" VARCHAR2(100),
	"TOPIC_KEYWORDS" VARCHAR2(100),
	"TOPIC_DESC" VARCHAR2(100)
   );
ALTER TABLE "STAPUSER"."STAP_NPS_TOPICS_ALL" ADD (CONSTRAINT topic_seq PRIMARY KEY (NPS_TOPIC_ID));

DROP TABLE "STAPUSER"."STAP_NPS_TOPICS_COMMENTS_H";
CREATE TABLE "STAPUSER"."STAP_NPS_TOPICS_COMMENTS_H"
   (	"NPS_TOPIC_ID" VARCHAR2(100) NOT NULL ENABLE,
	"COMM_ID" VARCHAR2(100) NOT NULL ENABLE,
	"TOPIC_PERC_CONTRIB" NUMBER(18,6)
   );